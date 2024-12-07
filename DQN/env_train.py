import os
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure



import numpy as np
import gym
from gym import spaces
import random

class ATCSchedulingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ATCSchedulingEnv, self).__init__()

        # Set random seed for reproducibility
        self.seed(42)

        # Controller and Role Setup
        self.num_controllers = 100
        self.num_roles = 7  # Modified from 8 to 7 roles
        self.time_slots_per_day = 16  # Modified from 24 to 16 time slots per day
        self.days_per_week = 5        # Modified from 7 to 5 days per week
        self.num_time_slots = self.time_slots_per_day * self.days_per_week  # Total time slots: 16 * 5 = 80

        # Constraints
        self.max_hours_per_week = 48
        self.max_hours_per_day = 8
        self.max_continuous_hours = 2
        self.min_break_slots = 1

        # Action space (Discrete)
        self.action_space = spaces.Discrete(self.num_controllers)

        # Observation space
        self.observation_space = spaces.Dict({
            'controller_states': spaces.Box(
                low=0,
                high=1,  # Normalized between 0 and 1
                shape=(self.num_controllers, 7),  # Adjusted shape to accommodate new state variables
                dtype=np.float32
            ),
            'current_time_slot': spaces.Box(
                low=0,
                high=1,  # Normalized between 0 and 1
                shape=(1,),
                dtype=np.float32
            ),
            'current_role': spaces.Box(
                low=0,
                high=self.num_roles - 1,
                shape=(1,),
                dtype=np.int32
            )
        })

        # Initialize certifications as a boolean array
        self.controller_certifications = np.zeros((self.num_controllers, self.num_roles), dtype=bool)

        # Initialize state variables
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # Controller states: [Total Hours Worked, Continuous Hours Worked, Time Since Last Shift,
        # Hours Worked This Week, Hours Worked Today, Earliest Assignment Time Slot in Day, Latest Assignment Time Slot in Day]
        self.controller_states = np.zeros((self.num_controllers, 7), dtype=np.float32)
        self.controller_states[:, 2] = self.min_break_slots  # Time Since Last Shift
        self.controller_states[:, 5] = self.time_slots_per_day  # Earliest Assignment Time Slot in Day
        self.controller_states[:, 6] = -1  # Latest Assignment Time Slot in Day

        self.current_time_slot = 0
        self.current_role_idx = 0  # Start with the first role
        self.done = False
        self.controllers_used = set()
        self.total_negative_reward = 0.0  # Accumulate penalties over time
        self.invalid_actions_count = 0

        # Initialize schedule_grid with roles as rows and time slots as columns
        self.schedule_grid = np.full((self.num_roles, self.num_time_slots), -1, dtype=int)

        # Initialize controller certifications
        num_certified_roles_per_controller = 5  # Adjust as needed to ensure feasibility
        for i in range(self.num_controllers):
            certified_roles = np.random.choice(self.num_roles, num_certified_roles_per_controller, replace=False)
            self.controller_certifications[i, certified_roles] = True

        return self._get_observation()

    def _get_observation(self):
        # Normalize controller_states
        max_values = np.array([
            self.num_time_slots,           # Total Hours Worked
            self.max_continuous_hours,     # Continuous Hours Worked
            self.num_time_slots,           # Time Since Last Shift
            self.max_hours_per_week,       # Hours Worked This Week
            self.max_hours_per_day,        # Hours Worked Today
            self.time_slots_per_day - 1,   # Earliest Assignment Time Slot in Day
            self.time_slots_per_day - 1    # Latest Assignment Time Slot in Day
        ], dtype=np.float32)

        normalized_controller_states = self.controller_states / max_values

        # Normalize current_time_slot
        normalized_time_slot = np.array([self.current_time_slot / (self.num_time_slots - 1)], dtype=np.float32)

        observation = {
            'controller_states': normalized_controller_states,
            'current_time_slot': normalized_time_slot,
            'current_role': np.array([self.current_role_idx], dtype=np.int32)
        }

        return observation

    def step(self, action):
        reward = 0.0
        info = {}
        invalid_action = False

        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action = action.item()
        controller_idx = int(action)
        role_idx = self.current_role_idx

        # Check if action is valid
        if not self._is_action_valid(controller_idx, role_idx):
            reward = -10  # Penalty for invalid action
            self.invalid_actions_count += 1
            invalid_action = True
        else:
            # Assign controller to role
            self._assign_controller(controller_idx, role_idx)
            reward = 1.0  # Positive reward for valid assignment

        # Move to next role or time slot
        self.current_role_idx += 1
        if self.current_role_idx >= self.num_roles:
            # Move to next time slot
            self.current_role_idx = 0
            self.current_time_slot += 1

            # Reset weekly hours at the start of the week
            if self.current_time_slot % (self.time_slots_per_day * self.days_per_week) == 0 and self.current_time_slot != 0:
                self.controller_states[:, 3] = 0  # Reset Hours Worked This Week

            # Reset daily hours at the start of each day
            if self.current_time_slot % self.time_slots_per_day == 0 and self.current_time_slot != 0:
                self.controller_states[:, 4] = 0  # Reset Hours Worked Today
                self.controller_states[:, 5] = self.time_slots_per_day  # Reset Earliest Assignment Time Slot in Day
                self.controller_states[:, 6] = -1  # Reset Latest Assignment Time Slot in Day

            # Update Time Since Last Shift and reset Continuous Hours Worked for all controllers
            self.controller_states[:, 2] += 1  # Time Since Last Shift
            self.controller_states[:, 1] = 0   # Reset Continuous Hours Worked

        if self.current_time_slot >= self.num_time_slots:
            self.done = True
            # Add info for logging
            info['total_controllers_used'] = len(self.controllers_used)
            info['invalid_actions'] = self.invalid_actions_count
        else:
            self.done = False

        observation = self._get_observation()
        return observation, reward, self.done, info

    def _is_action_valid(self, controller_idx, role_idx):
        # Check if controller is certified for the role
        if not self.controller_certifications[controller_idx, role_idx]:
            return False

        # Check if controller is already assigned in this time slot
        if controller_idx in self.schedule_grid[:, self.current_time_slot]:
            return False

        # Check constraints
        controller_state = self.controller_states[controller_idx]

        # Exceeding weekly hours
        if controller_state[3] >= self.max_hours_per_week:
            return False

        # Exceeding daily hours
        if controller_state[4] >= self.max_hours_per_day:
            return False

        # Exceeding continuous hours without break
        if controller_state[1] >= self.max_continuous_hours and controller_state[2] < self.min_break_slots:
            return False

        # Insufficient break
        if controller_state[2] < self.min_break_slots and controller_state[1] > 0:
            return False

        # Exceeding daily span limit (gap between earliest and latest assignment > 8 slots)
        current_time_slot_in_day = self.current_time_slot % self.time_slots_per_day
        potential_earliest = min(controller_state[5], current_time_slot_in_day)
        potential_latest = max(controller_state[6], current_time_slot_in_day)
        potential_gap = potential_latest - potential_earliest + 1
        if potential_gap > 8:
            return False

        return True

    def _assign_controller(self, controller_idx, role_idx):
        current_time_slot_in_day = self.current_time_slot % self.time_slots_per_day
        controller_state = self.controller_states[controller_idx]

        # Update controller state
        controller_state[0] += 1  # Total Hours Worked
        controller_state[1] += 1  # Continuous Hours Worked
        controller_state[2] = 0   # Time Since Last Shift reset
        controller_state[3] += 1  # Hours Worked This Week
        controller_state[4] += 1  # Hours Worked Today

        # Update Earliest and Latest Assignment Time Slot in Day
        controller_state[5] = min(controller_state[5], current_time_slot_in_day)
        controller_state[6] = max(controller_state[6], current_time_slot_in_day)

        # Store the role assignment in the schedule grid
        self.schedule_grid[role_idx, self.current_time_slot] = controller_idx

        # Add to controllers used
        self.controllers_used.add(controller_idx)

    def render(self, mode='human'):
        # Rendering not implemented
        pass

    def close(self):
        pass







import os
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

# Define the directory for checkpoints and logs
checkpoint_dir = './ATC_Scheduling/Checkpoints'
tensorboard_log_dir = './ATC_Scheduling/TensorBoard'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Custom callback for real-time logging during training
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)
        self.writer = writer
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # Accumulate rewards for the current episode
        if "reward" in self.locals:
            reward = self.locals["rewards"]
            self.episode_rewards.append(reward)

        return True

    def _on_rollout_end(self):
        # Calculate and log mean reward for the rollout
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards)
            self.writer.add_scalar("Rollout/Mean_Reward", mean_reward, self.num_timesteps)
            self.episode_rewards = []  # Reset for the next rollout

        # Log other statistics if available
        if "infos" in self.locals:
            for key, value in self.locals["infos"][0].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Infos/{key}", value, self.num_timesteps)

    def _on_training_end(self):
        # Finalize logging
        self.writer.flush()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Create the environment
env = ATCSchedulingEnv()
env = make_vec_env(lambda: env, n_envs=1)

# Create the PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=tensorboard_log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    ent_coef=0.01,
    n_epochs=10,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
)

# Create the checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=200000,
    save_path=checkpoint_dir,
    name_prefix='ppo_atc_scheduling'
)

# Create TensorBoard logging callback
tensorboard_callback = TensorboardLoggingCallback(writer=writer)

# Train the model with both callbacks
model.learn(
    total_timesteps=5000000,
    callback=[checkpoint_callback, tensorboard_callback]
)

# Close the writer after training
writer.close()


