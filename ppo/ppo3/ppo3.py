import numpy as np
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.distributions import MultiCategoricalDistribution

class ATCSchedulingEnv_b(gym.Env):
    """
    Custom Gym environment for Air Traffic Controller scheduling.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ATCSchedulingEnv_b, self).__init__()

        # Set random seed for reproducibility
        self.seed(42)

        # Controller and Position Setup
        self.num_controllers = 26  # Increased to 26
        self.num_positions = 7
        self.num_days = 7
        self.time_slots_per_day = 16  # Each time slot is 1.5 hours
        self.total_time_slots = self.num_days * self.time_slots_per_day  # 112 time slots

        # Duty Limits
        self.max_continuous_duty_hours = 12
        self.max_weekly_hours = 48
        self.max_monthly_hours = 190
        self.max_consecutive_days = 6
        self.mandatory_break_after_6_days = 2  # days
        self.break_after_night_duty = 2  # days
        self.min_break_between_shifts = 12  # hours

        # Penalties and their weights (revised)
        self.penalties = {
            'UNASSIGNED_POSITION': -50,
            'INVALID_ACTION': -100,
            'DUPLICATE_CONTROLLER': -20,
            'CONTINUOUS_12_HOURS': -100,
            'WEEKLY_HOURS': -50,
            'MONTHLY_HOURS': -50,
            'CONSECUTIVE_DAYS': -40,
            'MANDATORY_BREAK': -30,
            'NIGHT_DUTY_BREAK': -30,
            'MINIMUM_BREAK': -20,
            'NO_PENALTY': 0  # Ideal reward when no penalties are incurred
        }

        # Controller Requirements
        self.controller_medical_validity = np.ones(self.num_controllers, dtype=bool)
        self.controller_english_proficiency = np.ones(self.num_controllers, dtype=bool)
        self.controller_recency = np.ones((self.num_controllers, self.num_positions), dtype=bool)

        # Holidays and availability
        self.controller_holidays = {c: [] for c in range(self.num_controllers)}  # dict with list of days
        # Example: Controller 0 is on holiday on days 1, 3, and 4
        self.controller_holidays[0] = [1, 3, 4]

        # Initialize schedule grid: (num_days, time_slots_per_day, num_positions)
        self.schedule_grid = np.full((self.num_days, self.time_slots_per_day, self.num_positions), -1, dtype=int)

        # Controller states: [Total Hours Worked, Continuous Hours Worked, Time Since Last Shift, Fatigue Level,
        #                     Hours Worked This Week, Hours Worked Today]
        self.controller_states = np.zeros((self.num_controllers, 6), dtype=np.float32)

        # Action space: Include 'no controller' option (last index)
        self.action_space = spaces.MultiDiscrete([self.num_controllers + 1] * self.num_positions)

        # Observation space
        self.observation_space = spaces.Dict({
            'controller_states': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_controllers, 6),
                dtype=np.float32
            ),
            'current_time_slot': spaces.Box(
                low=0,
                high=self.time_slots_per_day - 1,
                shape=(1,),
                dtype=np.int32
            ),
            'current_day': spaces.Box(
                low=0,
                high=self.num_days - 1,
                shape=(1,),
                dtype=np.int32
            ),
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(self.num_positions, self.num_controllers + 1),
                dtype=np.bool_
            )
        })

        # Initialize state variables
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        # Save the current schedule grid before resetting
        if hasattr(self, 'schedule_grid'):
            self.last_schedule_grid = self.schedule_grid.copy()

        # Reset controller states
        self.controller_states = np.zeros((self.num_controllers, 6), dtype=np.float32)
        self.controller_states[:, 2] = self.min_break_between_shifts  # Time Since Last Shift

        self.current_time_slot = 0
        self.current_day = 0
        self.done = False
        self.controllers_used = set()
        self.total_negative_reward = 0.0  # Accumulate penalties over time
        self.invalid_actions_count = 0
        self.episode_reward = 0.0  # Initialize episode reward

        # Reset schedule grid
        self.schedule_grid = np.full((self.num_days, self.time_slots_per_day, self.num_positions), -1, dtype=int)

        # Generate initial action mask
        self.action_mask = self._generate_action_mask()

        return self._get_observation()

    def _get_observation(self):
        # Normalize controller_states
        max_values = np.array([
            self.total_time_slots * 1.5,  # Total Hours Worked
            self.max_continuous_duty_hours,  # Continuous Hours Worked
            self.total_time_slots * 1.5,  # Time Since Last Shift
            10,  # Fatigue Level
            self.max_weekly_hours,  # Hours Worked This Week
            self.max_continuous_duty_hours  # Hours Worked Today
        ], dtype=np.float32)

        normalized_controller_states = self.controller_states / max_values

        observation = {
            'controller_states': normalized_controller_states,
            'current_time_slot': np.array([self.current_time_slot], dtype=np.int32),
            'current_day': np.array([self.current_day], dtype=np.int32),
            'action_mask': self.action_mask.copy()
        }

        return observation

    def step(self, action):
        reward = 0.0
        info = {}
        total_negative_reward = 0

        role_assignments = action  # Controllers assigned to positions at current time slot
        controllers_assigned = set()

        for position_idx, controller_idx in enumerate(role_assignments):
            if controller_idx == self.num_controllers:
                # 'No controller' selected
                total_negative_reward += self.penalties['UNASSIGNED_POSITION']
                self.invalid_actions_count += 1
                self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = -1
                continue

            if not self.action_mask[position_idx, controller_idx]:
                total_negative_reward += self.penalties['INVALID_ACTION']
                self.invalid_actions_count += 1
                self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = -1
                continue

            if controller_idx in controllers_assigned:
                total_negative_reward += self.penalties['DUPLICATE_CONTROLLER']
                self.invalid_actions_count += 1
                self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = -1
                continue

            controllers_assigned.add(controller_idx)

            # Update controller state
            controller_state = self.controller_states[controller_idx]
            controller_state[0] += 1.5  # Total Hours Worked (1.5 hours per slot)
            controller_state[1] += 1.5  # Continuous Hours Worked
            controller_state[2] = 0     # Time Since Last Shift reset
            controller_state[3] += 1    # Fatigue Level increment
            controller_state[4] += 1.5  # Hours Worked This Week
            controller_state[5] += 1.5  # Hours Worked Today

            # Store the assignment in the schedule grid
            self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = controller_idx

            # Add to controllers used
            self.controllers_used.add(controller_idx)

            # Apply penalties for constraint violations
            if controller_state[1] > self.max_continuous_duty_hours:
                total_negative_reward += self.penalties['CONTINUOUS_12_HOURS']
                self.invalid_actions_count += 1

            if controller_state[4] > self.max_weekly_hours:
                total_negative_reward += self.penalties['WEEKLY_HOURS']
                self.invalid_actions_count += 1

            # Additional constraints can be added here

        # Update states for controllers not assigned
        all_controllers = np.arange(self.num_controllers)
        not_assigned_controllers = np.setdiff1d(all_controllers, np.array(list(controllers_assigned)))
        if len(not_assigned_controllers) > 0:
            # Time Since Last Shift
            self.controller_states[not_assigned_controllers, 2] += 1.5  # Increase by 1.5 hours
            # Reset Continuous Hours Worked if resting
            self.controller_states[not_assigned_controllers, 1] = 0
            # Reduce Fatigue if resting
            self.controller_states[not_assigned_controllers, 3] = np.maximum(
                self.controller_states[not_assigned_controllers, 3] - 1, 0
            )

        # Accumulate total negative reward
        self.total_negative_reward += total_negative_reward

        # Move to next time slot
        self.current_time_slot += 1
        if self.current_time_slot >= self.time_slots_per_day:
            self.current_time_slot = 0
            self.current_day += 1
            # Reset daily hours
            self.controller_states[:, 5] = 0  # Reset Hours Worked Today

        if self.current_day >= self.num_days:
            self.done = True
            # Calculate total reward
            total_reward = - self.total_negative_reward
            reward = total_reward  # Total reward is negative penalties

            # Add info for logging
            info['total_controllers_used'] = len(self.controllers_used)
            info['invalid_actions'] = self.invalid_actions_count
            info['total_negative_reward'] = self.total_negative_reward
        else:
            # Per-step reward is negative of the penalties incurred
            reward = -total_negative_reward

        self.episode_reward += reward  # Accumulate episode reward

        # Generate new action mask
        self.action_mask = self._generate_action_mask()

        observation = self._get_observation()
        return observation, reward, self.done, info

    def _generate_action_mask(self):
        # Generate action mask based on constraints
        mask = np.ones((self.num_positions, self.num_controllers + 1), dtype=bool)
        mask[:, self.num_controllers] = False  # Initially disable 'no controller' option

        # Apply medical validity and English proficiency
        invalid_medical = ~self.controller_medical_validity
        invalid_english = ~self.controller_english_proficiency
        for c in range(self.num_controllers):
            if invalid_medical[c] or invalid_english[c]:
                mask[:, c] = False

        # Apply recency
        for position_idx in range(self.num_positions):
            for controller_idx in range(self.num_controllers):
                if not self.controller_recency[controller_idx, position_idx]:
                    mask[position_idx, controller_idx] = False

        # Apply holidays
        current_day = self.current_day
        for c in range(self.num_controllers):
            if current_day in self.controller_holidays[c]:
                mask[:, c] = False

        # Apply duty limits
        over_weekly_hours = self.controller_states[:, 4] >= self.max_weekly_hours
        over_continuous_hours = self.controller_states[:, 1] >= self.max_continuous_duty_hours
        for c in range(self.num_controllers):
            if over_weekly_hours[c] or over_continuous_hours[c]:
                mask[:, c] = False

        # Additional constraints can be applied here

        # Enable 'no controller' action where necessary
        for position_idx in range(self.num_positions):
            if not np.any(mask[position_idx, :self.num_controllers]):
                mask[position_idx, self.num_controllers] = True

        return mask

    def render(self, mode='human'):
        print(f"Day {self.current_day + 1}, Time Slot {self.current_time_slot + 1}")
        print("Schedule Grid at current day and time slot:")
        print(self.schedule_grid[self.current_day, self.current_time_slot])

    def close(self):
        pass

# Define custom feature extractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        self.extractors = {}

        # Controller states
        controller_states_shape = observation_space.spaces['controller_states'].shape
        self.extractors['controller_states'] = nn.Sequential(
            nn.Flatten(),
            nn.Linear(controller_states_shape[0]*controller_states_shape[1], 256),
            nn.ReLU(),
        )

        # Current time slot
        n_time_slots = observation_space.spaces['current_time_slot'].high.item() + 1
        self.extractors['current_time_slot'] = nn.Embedding(
            n_time_slots, 32)

        # Current day
        n_days = observation_space.spaces['current_day'].high.item() + 1
        self.extractors['current_day'] = nn.Embedding(
            n_days, 32)

        # Compute the total features dimension
        total_concat_size = 256 + 32 + 32

        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # Controller states
        obs = observations['controller_states'].float()
        encoded = self.extractors['controller_states'](obs)
        encoded_tensor_list.append(encoded)

        # Current time slot
        obs = observations['current_time_slot'].long()
        obs = obs.squeeze(-1)  # Remove extra dimension if present
        encoded = self.extractors['current_time_slot'](obs)
        encoded_tensor_list.append(encoded)

        # Current day
        obs = observations['current_day'].long()
        obs = obs.squeeze(-1)  # Remove extra dimension if present
        encoded = self.extractors['current_day'](obs)
        encoded_tensor_list.append(encoded)

        return torch.cat(encoded_tensor_list, dim=1)

# Define custom policy with action masking
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={}
        )
        # Initialize the action distribution
        self.action_dist = MultiCategoricalDistribution(self.action_space.nvec)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_distribution(self, obs, latent_pi):
        # Get the logits for each discrete action space
        action_logits = self.action_net(latent_pi)
        split_sizes = self.action_space.nvec.tolist()
        action_logits = torch.split(action_logits, split_sizes, dim=1)

        # Get action masks from the observations
        action_masks = obs['action_mask']  # Shape: [batch_size, num_positions, num_controllers + 1]
        # Transpose to match the dimensions: [num_positions, batch_size, num_controllers + 1]
        action_masks = action_masks.permute(1, 0, 2)

        # Apply the masks to the logits
        masked_logits = []
        for logits, mask in zip(action_logits, action_masks):
            # Ensure mask and logits are float tensors
            mask = mask.float()
            logits = logits + (mask - 1) * 1e8  # Set logits of invalid actions to a large negative number
            masked_logits.append(logits)

        # Concatenate masked logits
        masked_logits = torch.cat(masked_logits, dim=1)

        # Create the MultiCategorical distribution with masked logits
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)
        return distribution

    def get_distribution(self, obs, latent_pi):
        return self._get_distribution(obs, latent_pi)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

# Custom callback to adjust entropy coefficient based on performance
class EntropyAdjustingCallback(BaseCallback):
    def __init__(self, verbose=0, min_episodes=10, increase_factor=1.1, decrease_factor=0.95,
                 max_ent_coef=0.1, min_ent_coef=0.0001):
        super(EntropyAdjustingCallback, self).__init__(verbose)
        self.min_episodes = min_episodes
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_ent_coef = max_ent_coef
        self.min_ent_coef = min_ent_coef
        self.episode_rewards = []
        self.previous_mean_reward = None

    def _on_step(self):
        if self.locals['dones'][0]:
            # Episode finished
            episode_reward = self.locals['infos'][0].get('episode')['r']
            self.episode_rewards.append(episode_reward)

            # Log the episode reward to TensorBoard
            self.logger.record('episode_reward', episode_reward)

            if len(self.episode_rewards) >= self.min_episodes:
                current_mean_reward = np.mean(self.episode_rewards[-self.min_episodes:])
                if self.previous_mean_reward is not None:
                    if current_mean_reward <= self.previous_mean_reward:
                        # Mean reward hasn't improved, increase entropy coefficient
                        new_ent_coef = min(self.model.ent_coef * self.increase_factor, self.max_ent_coef)
                        self.model.ent_coef = new_ent_coef
                        if self.verbose > 0:
                            print(f"Entropy coefficient increased to {self.model.ent_coef}")
                    else:
                        # Mean reward has improved, decrease entropy coefficient
                        new_ent_coef = max(self.model.ent_coef * self.decrease_factor, self.min_ent_coef)
                        self.model.ent_coef = new_ent_coef
                        if self.verbose > 0:
                            print(f"Entropy coefficient decreased to {self.model.ent_coef}")
                    # Log the entropy coefficient to TensorBoard
                    self.logger.record('entropy_coefficient', self.model.ent_coef)
                self.previous_mean_reward = current_mean_reward
        return True

# Custom callback to log the schedule every 10000 steps
class ScheduleLogger(BaseCallback):
    def __init__(self, log_freq=10000, log_dir='schedules', verbose=0):
        super(ScheduleLogger, self).__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            env = self.training_env.envs[0]
            # Access the last schedule grid
            if hasattr(env, 'last_schedule_grid'):
                schedule = env.last_schedule_grid
                filename = os.path.join(self.log_dir, f"schedule_step_{self.num_timesteps}.npy")
                np.save(filename, schedule)
        return True

# Create the environment
env = ATCSchedulingEnv_b()

# Create the PPO model with tensorboard logging
model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./ppo_atc_tensorboard/")

# Set up the callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='ppo_atc_scheduling')
schedule_logger = ScheduleLogger(log_freq=10000, log_dir='schedules')
entropy_adjusting_callback = EntropyAdjustingCallback(verbose=1)

# Combine callbacks
callback = CallbackList([checkpoint_callback, schedule_logger, entropy_adjusting_callback])

# Train the agent with the callbacks
model.learn(total_timesteps=100000000, callback=callback)

# Save the trained model
model.save("ppo_atc_scheduling_final")

# To test the trained model
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
