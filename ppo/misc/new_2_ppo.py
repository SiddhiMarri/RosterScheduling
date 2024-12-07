import numpy as np
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import torch.optim as optim
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.logger import configure

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
        self.num_controllers = 23
        self.num_positions = 7
        self.num_days = 7
        self.time_slots_per_day = 16  # Each time slot is 1.5 hours
        self.total_time_slots = self.num_days * self.time_slots_per_day  # 112 time slots

        # Shifts
        self.shifts = ['MORNING', 'AFTERNOON', 'NIGHT', 'NIGHT_OFF', 'CLEAR_OFF']
        self.shift_time_slots = {
            'MORNING': list(range(0, 4)),      # slots 0-3 (6 hours)
            'AFTERNOON': list(range(4, 8)),    # slots 4-7 (6.5 hours)
            'NIGHT': list(range(8, 16)),       # slots 8-15 (11 hours)
            'NIGHT_OFF': [],                   # No slots assigned
            'CLEAR_OFF': []                    # No slots assigned
        }

        # Teams and Team Cycle
        self.teams = ['A', 'B', 'C', 'D', 'E']
        self.team_cycle = ['MORNING', 'AFTERNOON', 'NIGHT', 'NIGHT_OFF', 'CLEAR_OFF']

        # Assign controllers to teams
        self.team_assignments = {
            'A': [0, 1, 2, 3, 4],
            'B': [5, 6, 7, 8, 9],
            'C': [10, 11, 12, 13],
            'D': [14, 15, 16, 17],
            'E': [18, 19, 20, 21, 22]
        }

        # Duty Limits
        self.max_continuous_duty_hours = 12
        self.max_weekly_hours = 48
        self.max_monthly_hours = 190
        self.max_consecutive_days = 6
        self.mandatory_break_after_6_days = 2  # days
        self.break_after_night_duty = 2  # days
        self.min_break_between_shifts = 12  # hours

        # Violations and their penalties/rewards
        self.penalties = {
            'CONTINUOUS_12_HOURS': -100,
            'WEEKLY_HOURS': -50,
            'MONTHLY_HOURS': -50,
            'CONSECUTIVE_DAYS': -40,
            'MANDATORY_BREAK': -30,
            'NIGHT_DUTY_BREAK': -30,
            'MINIMUM_BREAK': -20,
            'RECENCY': -10,
            'VALID_ACTION': 1,  # Positive reward for valid actions
            # Add other penalties/rewards as needed
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

        # Historical data for augmented observation space
        self.historical_assignments = np.zeros((self.num_controllers, self.num_positions), dtype=np.float32)

        # Action space: Assign controllers to positions for the current time slot on the current day
        self.action_space = spaces.MultiDiscrete([self.num_controllers] * self.num_positions)

        # Observation space
        self.observation_space = spaces.Dict({
            'controller_states': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_controllers, 6),
                dtype=np.float32
            ),
            'historical_assignments': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_controllers, self.num_positions),
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
                shape=(self.num_positions, self.num_controllers),
                dtype=np.bool_
            )
        })

        # Initialize state variables
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # Reset controller states
        self.controller_states = np.zeros((self.num_controllers, 6), dtype=np.float32)
        self.controller_states[:, 2] = self.min_break_between_shifts  # Time Since Last Shift

        self.historical_assignments = np.zeros((self.num_controllers, self.num_positions), dtype=np.float32)

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

        # Normalize historical assignments
        normalized_historical_assignments = self.historical_assignments / np.max(self.historical_assignments + 1e-8)

        observation = {
            'controller_states': normalized_controller_states,
            'historical_assignments': normalized_historical_assignments,
            'current_time_slot': np.array([self.current_time_slot], dtype=np.int32),
            'current_day': np.array([self.current_day], dtype=np.int32),
            'action_mask': self.action_mask.copy()
        }

        return observation

    def step(self, action):
        reward = 0.0
        info = {}
        invalid_action = False

        role_assignments = action  # Controllers assigned to positions at current time slot
        controllers_assigned = set()
        total_negative_reward = 0

        # Update controller states and schedule grid
        for position_idx, controller_idx in enumerate(role_assignments):
            if not self.action_mask[position_idx, controller_idx]:
                total_negative_reward += 10  # Penalty for invalid action
                self.invalid_actions_count += 1
                invalid_action = True
                continue

            if controller_idx in controllers_assigned:
                total_negative_reward += 10  # Penalty for assigning the same controller to multiple positions
                self.invalid_actions_count += 1
                invalid_action = True
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

            # Update historical assignments
            self.historical_assignments[controller_idx, position_idx] += 1

            # Store the assignment in the schedule grid
            self.schedule_grid[self.current_day, self.current_time_slot, position_idx] = controller_idx

            # Add to controllers used
            self.controllers_used.add(controller_idx)

            # Apply penalties for constraint violations
            if controller_state[1] > self.max_continuous_duty_hours:
                total_negative_reward += self.penalties['CONTINUOUS_12_HOURS']
                self.invalid_actions_count += 1
                invalid_action = True

            if controller_state[4] > self.max_weekly_hours:
                total_negative_reward += self.penalties['WEEKLY_HOURS']
                self.invalid_actions_count += 1
                invalid_action = True

            # Additional constraints can be added here

            # Positive reward for valid action
            if not invalid_action:
                reward += self.penalties['VALID_ACTION']

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
            total_reward = reward - self.total_negative_reward
            reward = total_reward / (self.num_positions * self.total_time_slots)  # Normalize reward

            # Add info for logging
            info['total_controllers_used'] = len(self.controllers_used)
            info['invalid_actions'] = self.invalid_actions_count
            info['total_negative_reward'] = self.total_negative_reward
        else:
            # Per-step reward includes positive rewards and penalties
            reward = reward - (total_negative_reward / self.num_positions)

        self.episode_reward += reward  # Accumulate episode reward

        # Generate new action mask
        self.action_mask = self._generate_action_mask()

        observation = self._get_observation()
        return observation, reward, self.done, info

    def _generate_action_mask(self):
        # Generate action mask based on constraints
        mask = np.ones((self.num_positions, self.num_controllers), dtype=bool)

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

        # Incorporate domain knowledge: Assign controllers only within their teams
        # For this example, we assume each team works specific shifts
        current_shift = self.team_cycle[self.current_day % len(self.team_cycle)]
        available_teams = [team for idx, team in enumerate(self.team_cycle) if team == current_shift]
        allowed_controllers = []
        for team in available_teams:
            allowed_controllers.extend(self.team_assignments[team])

        for c in range(self.num_controllers):
            if c not in allowed_controllers:
                mask[:, c] = False

        # Apply other constraints as needed

        return mask

    def render(self, mode='human'):
        print(f"Day {self.current_day + 1}, Time Slot {self.current_time_slot + 1}")
        print("Schedule Grid at current day and time slot:")
        print(self.schedule_grid[self.current_day, self.current_time_slot])

    def close(self):
        pass

# Define custom feature extractor with attention mechanism
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(AttentionFeatureExtractor, self).__init__(observation_space, features_dim=1)

        self.controller_states_shape = observation_space.spaces['controller_states'].shape
        self.num_controllers = self.controller_states_shape[0]
        self.state_dim = self.controller_states_shape[1]

        # Controller states embedding
        self.controller_embedding = nn.Linear(self.state_dim, 128)

        # Attention mechanism
        self.attention_layer = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        # Historical assignments embedding
        self.historical_embedding = nn.Linear(self.num_positions, 128)

        # Time slot embedding
        n_time_slots = observation_space.spaces['current_time_slot'].high.item() + 1
        self.time_slot_embedding = nn.Embedding(n_time_slots, 16)

        # Day embedding
        n_days = observation_space.spaces['current_day'].high.item() + 1
        self.day_embedding = nn.Embedding(n_days, 16)

        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 16 + 16, 256),
            nn.ReLU(),
        )

        # Compute the total features dimension
        self._features_dim = 256

    def forward(self, observations):
        # Controller states
        controller_states = observations['controller_states'].float()  # Shape: [batch_size, num_controllers, state_dim]
        batch_size = controller_states.shape[0]
        controller_states = controller_states.view(-1, self.state_dim)  # Shape: [batch_size * num_controllers, state_dim]
        controller_embedded = self.controller_embedding(controller_states)  # Shape: [batch_size * num_controllers, 128]
        controller_embedded = controller_embedded.view(batch_size, self.num_controllers, 128)  # [batch_size, num_controllers, 128]
        controller_embedded = controller_embedded.permute(1, 0, 2)  # [num_controllers, batch_size, 128]

        # Historical assignments
        historical_assignments = observations['historical_assignments'].float()  # [batch_size, num_controllers, num_positions]
        historical_assignments = historical_assignments.view(-1, self.num_positions)
        historical_embedded = self.historical_embedding(historical_assignments)  # [batch_size * num_controllers, 128]
        historical_embedded = historical_embedded.view(batch_size, self.num_controllers, 128)
        historical_embedded = historical_embedded.permute(1, 0, 2)  # [num_controllers, batch_size, 128]

        # Combine embeddings
        combined_embeddings = controller_embedded + historical_embedded  # [num_controllers, batch_size, 128]

        # Apply attention mechanism
        attn_output, _ = self.attention_layer(combined_embeddings, combined_embeddings, combined_embeddings)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, num_controllers, 128]
        attn_output = attn_output.reshape(batch_size, -1)  # [batch_size, num_controllers * 128]

        # Time slot embedding
        time_slot = observations['current_time_slot'].long().squeeze(-1)  # [batch_size]
        time_slot_embedded = self.time_slot_embedding(time_slot)  # [batch_size, 16]

        # Day embedding
        day = observations['current_day'].long().squeeze(-1)  # [batch_size]
        day_embedded = self.day_embedding(day)  # [batch_size, 16]

        # Concatenate all features
        features = torch.cat([attn_output, time_slot_embedded, day_embedded], dim=1)

        # Final layers
        output = self.fc(features)

        return output

# Define custom policy with action masking
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs={}
        )
        # Initialize the action distribution
        self.action_dist = MultiCategoricalDistribution(self.action_space.nvec)

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
        split_sizes = self.action_space.nvec.tolist()
        action_logits = torch.split(action_logits, split_sizes, dim=1)

        # Get action masks from the observations
        action_masks = obs['action_mask']  # Shape: [batch_size, num_positions, num_controllers]
        batch_size = action_masks.shape[0]
        num_positions = action_masks.shape[1]
        action_masks = action_masks.float()

        # Apply the masks to the logits
        masked_logits = []
        for i in range(num_positions):
            logits = action_logits[i]
            mask = action_masks[:, i, :]  # [batch_size, num_controllers]
            # Set logits of invalid actions to a large negative number
            logits = logits + (mask - 1) * 1e8
            masked_logits.append(logits)

        # Concatenate masked logits
        masked_logits = torch.cat(masked_logits, dim=1)

        # Create the MultiCategorical distribution with masked logits
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)
        return distribution

# Custom callback to log episode rewards to CSV
class EpisodeRewardLogger(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(EpisodeRewardLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_num = 0
        self.csv_file = open(f"{self.log_dir}/episode_rewards.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Episode', 'Reward'])  # Header
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate rewards
        self.episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_num += 1
            self.csv_writer.writerow([self.episode_num, self.episode_reward])
            self.episode_reward = 0.0
        return True

    def _on_training_end(self):
        # Close the CSV file when training ends
        self.csv_file.close()

# Configure TensorBoard logger
log_dir = "./tensorboard_logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

if __name__ == '__main__':
    # Create the environment
    env = ATCSchedulingEnv_b()

    # Create the PPO model
    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Set up the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./checkpoints/',
        name_prefix='ppo_atc_scheduling'
    )

    # Set up the episode reward logger callback
    reward_logger = EpisodeRewardLogger(log_dir='.')

    # Evaluation callback
    eval_env = ATCSchedulingEnv_b()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model',
        log_path='./eval_logs',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, reward_logger, eval_callback])

    # Train the agent with the callbacks
    model.learn(total_timesteps=100000, callback=callback)

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
