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
    Custom Gym environment for Air Traffic Controller scheduling with optimized performance.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ATCSchedulingEnv_b, self).__init__()

        # Set random seed for reproducibility
        self.seed(42)

        # Controller and Position Setup
        self.num_controllers = 200  # Number of controllers
        self.num_positions = 7
        self.num_days = 7
        self.time_slots_per_day = 24  # Each time slot is 1 hour
        self.total_time_slots = self.num_days * self.time_slots_per_day  # 168 time slots

        # Duty Limits and Breaks
        self.max_continuous_work_slots = 2  # Max continuous work sessions in slots (2 slots)
        self.mandatory_break_slots = 1  # Mandatory break slots after max continuous work
        self.max_daily_duty_slots = 12  # Max duty slots per day (12 hours)
        self.max_weekly_duty_slots = 48  # Max duty slots per week (48 hours)

        # Penalties and Rewards
        self.penalties = {
            'EXCEED_CONTINUOUS_WORK': -100,
            'NO_MANDATORY_BREAK': -80,
            'DAILY_DUTY_LIMIT': -50,
            'WEEKLY_DUTY_LIMIT': -50,
            'INVALID_ACTION': -10,
            'DUPLICATE_ASSIGNMENT': -10,
            'UNASSIGNED_POSITION': -5,
            'HOLIDAY_VIOLATION': -20
        }

        self.rewards = {
            'SUCCESSFUL_ASSIGNMENT': +1
        }

        # Holidays and availability
        self.controller_holidays = {c: [] for c in range(self.num_controllers)}  # dict with list of days
        # Example: Controller 0 is on holiday on days 1, 3, and 4
        self.controller_holidays[0] = [1, 3, 4]

        # Initialize schedule grid: (num_days, time_slots_per_day, num_positions)
        self.schedule_grid = np.full((self.num_days, self.time_slots_per_day, self.num_positions), -1, dtype=int)

        # Controller states: [Total Hours Worked, Continuous Work Slots, Continuous Break Slots,
        #                     Hours Worked This Week, Hours Worked Today]
        self.controller_states = np.zeros((self.num_controllers, 5), dtype=np.float32)

        # Action space: Assign controllers to positions for the current time slot
        # Added 1 to num_controllers to represent the "no controller" option
        self.action_space = spaces.MultiDiscrete([self.num_controllers + 1] * self.num_positions)

        # Observation space
        self.observation_space = spaces.Dict({
            'controller_states': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_controllers, 5),
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
            )
        })

        # Initialize state variables
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        # Reset controller states
        self.controller_states = np.zeros((self.num_controllers, 5), dtype=np.float32)

        self.current_time_slot = 0
        self.current_day = 0
        self.done = False
        self.controllers_used = set()
        self.invalid_actions_count = 0
        self.episode_reward = 0.0  # Initialize episode reward

        # Reset schedule grid
        self.schedule_grid = np.full((self.num_days, self.time_slots_per_day, self.num_positions), -1, dtype=int)

        return self._get_observation()

    def _get_observation(self):
        # Normalize controller_states
        max_values = np.array([
            self.total_time_slots,          # Total Hours Worked
            self.max_continuous_work_slots, # Continuous Work Slots
            self.max_continuous_work_slots, # Continuous Break Slots (max possible)
            self.max_weekly_duty_slots,     # Hours Worked This Week
            self.max_daily_duty_slots       # Hours Worked Today
        ], dtype=np.float32)

        normalized_controller_states = self.controller_states / max_values

        observation = {
            'controller_states': normalized_controller_states,
            'current_time_slot': np.array([self.current_time_slot], dtype=np.int32),
            'current_day': np.array([self.current_day], dtype=np.int32)
        }

        return observation

    def step(self, action):
        reward = 0.0
        info = {}
        violation_count = 0

        role_assignments = action  # Controllers assigned to positions at current time slot
        controllers_assigned = set()

        # Update controller states and schedule grid
        for position_idx, controller_idx in enumerate(role_assignments):
            if controller_idx == self.num_controllers:
                # Agent chose to leave the position unassigned
                reward += self.penalties['UNASSIGNED_POSITION']
                violation_count += 1
                continue

            if controller_idx in controllers_assigned:
                reward += self.penalties['DUPLICATE_ASSIGNMENT']
                self.invalid_actions_count += 1
                violation_count += 1
                continue

            # Check if controller is available
            if self.current_day in self.controller_holidays[controller_idx]:
                reward += self.penalties['HOLIDAY_VIOLATION']
                violation_count += 1
                continue

            controllers_assigned.add(controller_idx)

            # Update controller state
            controller_state = self.controller_states[controller_idx]
            controller_state[0] += 1  # Total Hours Worked (1 hour per slot)
            controller_state[1] += 1  # Continuous Work Slots
            controller_state[2] = 0   # Reset Continuous Break Slots
            controller_state[3] += 1  # Hours Worked This Week
            controller_state[4] += 1  # Hours Worked Today

            # Store the assignment in the schedule grid
            self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = controller_idx

            # Add to controllers used
            self.controllers_used.add(controller_idx)

            # Apply penalties for constraint violations
            constraints_violated = False

            # Exceeding maximum continuous work slots
            if controller_state[1] > self.max_continuous_work_slots:
                reward += self.penalties['EXCEED_CONTINUOUS_WORK']
                violation_count += 1
                constraints_violated = True

            # Exceeding daily duty limit
            if controller_state[4] > self.max_daily_duty_slots:
                reward += self.penalties['DAILY_DUTY_LIMIT']
                violation_count += 1
                constraints_violated = True

            # Exceeding weekly duty limit
            if controller_state[3] > self.max_weekly_duty_slots:
                reward += self.penalties['WEEKLY_DUTY_LIMIT']
                violation_count += 1
                constraints_violated = True

            # If no constraints violated, give positive reward
            if not constraints_violated:
                reward += self.rewards['SUCCESSFUL_ASSIGNMENT']

        # Update states for controllers not assigned
        all_controllers = np.arange(self.num_controllers)
        not_assigned_controllers = np.setdiff1d(all_controllers, np.array(list(controllers_assigned)))
        if len(not_assigned_controllers) > 0:
            # Increase Continuous Break Slots
            self.controller_states[not_assigned_controllers, 2] += 1  # Increase by 1 hour
            # Reset Continuous Work Slots if resting
            self.controller_states[not_assigned_controllers, 1] = 0

            # Check for mandatory break after working max_continuous_work_slots
            for controller_idx in not_assigned_controllers:
                # If the controller had just worked max_continuous_work_slots and hasn't taken a mandatory break
                if self.controller_states[controller_idx, 1] == 0 and self.controller_states[controller_idx, 2] < self.mandatory_break_slots:
                    reward += self.penalties['NO_MANDATORY_BREAK']
                    violation_count += 1

        # Move to next time slot
        self.current_time_slot += 1
        if self.current_time_slot >= self.time_slots_per_day:
            self.current_time_slot = 0
            self.current_day += 1
            # Reset daily hours
            self.controller_states[:, 4] = 0  # Reset Hours Worked Today

        # Check if episode is done
        if self.current_day >= self.num_days:
            self.done = True
            info['total_controllers_used'] = len(self.controllers_used)
            info['invalid_actions'] = self.invalid_actions_count
            info['violations'] = violation_count
        else:
            self.done = False

        observation = self._get_observation()
        return observation, reward, self.done, info

    def render(self, mode='human'):
        print(f"Day {self.current_day + 1}, Time Slot {self.current_time_slot + 1}")
        print("Schedule Grid at current day and time slot:")
        print(self.schedule_grid[self.current_day, self.current_time_slot])

    def close(self):
        pass

# Define custom feature extractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=160)  # Adjusted features_dim

        self.extractors = {}

        # Controller states
        controller_states_shape = observation_space.spaces['controller_states'].shape
        self.extractors['controller_states'] = nn.Sequential(
            nn.Flatten(),
            nn.Linear(controller_states_shape[0]*controller_states_shape[1], 128),
            nn.ReLU(),
        )

        # Current time slot
        n_time_slots = observation_space.spaces['current_time_slot'].high.item() + 1
        self.extractors['current_time_slot'] = nn.Embedding(
            n_time_slots, 16)

        # Current day
        n_days = observation_space.spaces['current_day'].high.item() + 1
        self.extractors['current_day'] = nn.Embedding(
            n_days, 16)

        # Compute the total features dimension
        total_concat_size = 128 + 16 + 16

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
        # Re-initialize the action_net to ensure correct output size
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, int(sum(self.action_space.nvec)))

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self.get_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_distribution(self, obs, latent_pi):
        # Get the logits for each discrete action space
        action_logits = self.action_net(latent_pi)
        # The action_dist will split the action_logits internally
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution

# Custom callback to log episode rewards and violations to CSV
class EpisodeRewardLogger(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(EpisodeRewardLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_num = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.log_dir, "episode_rewards.csv"), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Episode', 'Reward', 'Violations'])  # Header

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_num += 1
            reward = self.locals['rewards'][0]
            info = self.locals['infos'][0]
            violations = info.get('violations', 0)
            self.csv_writer.writerow([self.episode_num, reward, violations])
            if self.verbose > 0:
                print(f"Episode {self.episode_num} Reward: {reward}, Violations: {violations}")
        return True

    def _on_training_end(self):
        # Close the CSV file when training ends
        self.csv_file.close()

# Custom callback to log the schedule every 1000 steps
class ScheduleLogger(BaseCallback):
    def __init__(self, log_freq=1000, log_dir='schedules', verbose=0):
        super(ScheduleLogger, self).__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_logged_timestep = 0  # Keep track of the last logged timestep

    def _on_step(self):
        if (self.num_timesteps - self.last_logged_timestep) >= self.log_freq:
            # Access the schedule grid from the environment
            schedule = self.training_env.get_attr('schedule_grid')[0].copy()
            filename = os.path.join(self.log_dir, f"schedule_step_{self.num_timesteps}.npy")
            np.save(filename, schedule)
            if self.verbose > 0:
                print(f"Saved schedule at timestep {self.num_timesteps} to {filename}")
            self.last_logged_timestep = self.num_timesteps  # Update the last logged timestep
        return True

if __name__ == "__main__":
    # Create the environment
    env = ATCSchedulingEnv_b()

    # Adjust training parameters
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[256, 256]
    )

    # Create the PPO model with adjusted parameters
    model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./ppo_atc_tensorboard/",
                learning_rate=3e-4, batch_size=64, n_steps=2048, policy_kwargs=policy_kwargs)

    # Set up the checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                             name_prefix='ppo_atc_scheduling')

    # Set up the episode reward logger callback
    reward_logger = EpisodeRewardLogger(log_dir='.')

    # Set up the schedule logger callback
    schedule_logger = ScheduleLogger(log_freq=10000, log_dir='schedules')

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, reward_logger, schedule_logger])

    # Train the agent with the callbacks
    model.learn(total_timesteps=10000000, callback=callback)

    # Save the trained model
    model.save("ppo_atc_scheduling_final")
