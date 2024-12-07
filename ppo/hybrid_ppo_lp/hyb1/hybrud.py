# Import necessary libraries
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing
import os
from datetime import datetime
import pulp

# Define the scheduling environment


class ATCSchedulingEnv(gym.Env):
    def __init__(self):
        super(ATCSchedulingEnv, self).__init__()

        # Define constants
        self.num_controllers = 100
        self.num_positions = 10
        self.num_days = 7
        self.time_slots_per_day = 24
        self.total_time_slots = self.num_days * self.time_slots_per_day

        # Action space: Assign a controller to each position
        self.action_space = spaces.MultiDiscrete(
            [self.num_controllers] * self.num_positions)

        # Observation space: Controller states
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_controllers, 7),  # 7 state features per controller
            dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def reset(self):
        # Initialize controller states
        self.controller_states = np.zeros(
            (self.num_controllers, 7), dtype=np.float32)
        # State features:
        # 0: Total Hours Worked Today (normalized)
        # 1: Total Hours Worked This Week (normalized)
        # 2: Total Hours Worked This Month (normalized)
        # 3: Consecutive Duty Days (normalized)
        # 4: Remaining Break Time (normalized)
        # 5: Worked Today Flag (0 or 1)
        # 6: Fatigue Level (normalized)

        # Initialize scheduling grid
        self.schedule_grid = np.full(
            (self.total_time_slots, self.num_positions), -1, dtype=np.int32)

        # Time variables
        self.current_time_slot = 0
        self.current_day = 0

        # Initialize done flag
        self.done = False

        return self._get_observation()

    def step(self, action):
        # Assign controllers to positions
        assignments = action

        # Validate assignments
        valid, penalty = self._validate_assignments(assignments)

        # Update controller states
        self._update_controller_states(assignments)

        # Save assignments to schedule grid
        self.schedule_grid[self.current_time_slot] = assignments

        # Compute reward
        reward = self._compute_reward(penalty)

        # Advance time
        self.current_time_slot += 1
        if self.current_time_slot % self.time_slots_per_day == 0:
            self.current_day += 1
            self._reset_daily_counters()

        if self.current_time_slot >= self.total_time_slots:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        return self.controller_states.copy()

    def _validate_assignments(self, assignments):
        penalty = 0

        # Constraint 1: Duplicate Assignment
        if len(set(assignments)) < len(assignments):
            penalty += 10  # Penalty for duplicate assignments

        # Constraint 2: Controller Availability and Breaks
        for idx, controller_id in enumerate(assignments):
            if controller_id == -1:
                penalty += 5  # Penalty for unfilled position
                continue

            controller_state = self.controller_states[controller_id]

            # Continuous Work Slots
            if controller_state[0] >= 2 / 12:
                penalty += 5  # Exceeded continuous work slots

            # Maximum Daily Duty Hours
            if controller_state[0] >= 12 / 12:
                penalty += 10  # Exceeded daily duty hours

            # Maximum Weekly Duty Hours
            if controller_state[1] >= 48 / (self.num_days * 12):
                penalty += 10  # Exceeded weekly duty hours

            # Maximum Monthly Duty Hours
            if controller_state[2] >= 190 / (30 * 12):
                penalty += 10  # Exceeded monthly duty hours

            # Maximum Consecutive Duty Days
            if controller_state[3] >= 6 / 6:
                penalty += 10  # Exceeded consecutive duty days

            # Break After Consecutive Days
            if controller_state[4] > 0:
                penalty += 5  # Should be on break

        valid = penalty == 0
        return valid, penalty

    def _update_controller_states(self, assignments):
        for controller_id in range(self.num_controllers):
            if controller_id in assignments:
                # Update working hours
                # Total Hours Worked Today
                self.controller_states[controller_id][0] += 1 / 12
                self.controller_states[controller_id][1] += 1 / \
                    (self.num_days * 12)  # Total Hours This Week
                # Total Hours This Month
                self.controller_states[controller_id][2] += 1 / (30 * 12)

                # Update consecutive duty days
                if self.controller_states[controller_id][5] == 0:
                    # Consecutive Duty Days
                    self.controller_states[controller_id][3] += 1 / 6

                # Set worked today flag
                self.controller_states[controller_id][5] = 1

                # Decrease break time if any
                if self.controller_states[controller_id][4] > 0:
                    self.controller_states[controller_id][4] -= 1 / 12

                # Update fatigue level (example increment)
                # Fatigue Level
                self.controller_states[controller_id][6] += 0.01
            else:
                # Mandatory Break Enforcement
                if self.controller_states[controller_id][0] >= 2 / 12:
                    # Remaining Break Time
                    self.controller_states[controller_id][4] += 1 / 12

                # Reset continuous work slots if on break
                if self.controller_states[controller_id][4] > 0:
                    # Reset Hours Worked Today
                    self.controller_states[controller_id][0] = 0

        # Normalize states
        self.controller_states = np.clip(self.controller_states, 0, 1)

    def _reset_daily_counters(self):
        for controller_id in range(self.num_controllers):
            # Reset daily hours
            # Total Hours Worked Today
            self.controller_states[controller_id][0] = 0

            # Reset worked today flag
            self.controller_states[controller_id][5] = 0

            # Check for 48-hour break after 6 consecutive days
            if self.controller_states[controller_id][3] >= 1:
                self.controller_states[controller_id][4] = 48 / \
                    (self.num_days * 12)  # Remaining Break Time
                # Reset Consecutive Duty Days
                self.controller_states[controller_id][3] = 0

    def _compute_reward(self, penalty):
        # Base reward for filling all positions
        reward = 10 - penalty
        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Define a callback to save the schedule grid periodically


class SaveScheduleCallback(BaseCallback):
    def __init__(self, env, check_freq, save_path, verbose=1):
        super(SaveScheduleCallback, self).__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            np.save(os.path.join(self.save_path,
                    f'schedule_{self.n_calls}.npy'), self.env.schedule_grid)
        return True

# Define the optimization function


def optimize_schedule(assignments, controller_states, env):
    # Create the LP problem
    prob = pulp.LpProblem("ATC_Scheduling", pulp.LpMinimize)

    # Decision variables
    x = {}
    for c in range(env.num_controllers):
        for p in range(env.num_positions):
            var_name = f'x_{c}_{p}'
            x[(c, p)] = pulp.LpVariable(var_name, cat='Binary')

    # Objective: Minimize the number of controllers used
    prob += pulp.lpSum(x[(c, p)] for c in range(env.num_controllers)
                       for p in range(env.num_positions))

    # Constraints
    # Each position must be filled by exactly one controller
    for p in range(env.num_positions):
        prob += pulp.lpSum(x[(c, p)] for c in range(env.num_controllers)) == 1

    # A controller cannot be assigned to more than one position simultaneously
    for c in range(env.num_controllers):
        prob += pulp.lpSum(x[(c, p)] for p in range(env.num_positions)) <= 1

    # Solve the problem
    prob.solve()

    # Extract the assignments
    optimized_assignments = [-1] * env.num_positions
    for c in range(env.num_controllers):
        for p in range(env.num_positions):
            if pulp.value(x[(c, p)]) == 1:
                optimized_assignments[p] = c

    return optimized_assignments

# Main training loop


def train_model():
    env = ATCSchedulingEnv()

    # Create save directory
    save_path = './schedules'
    os.makedirs(save_path, exist_ok=True)

    # Create the RL model
    model = PPO('MlpPolicy', env, verbose=1)

    # Create the callback
    callback = SaveScheduleCallback(
        env, check_freq=10000, save_path=save_path)

    # Train the model
    model.learn(total_timesteps=10000000, callback=callback)

    # Save the final schedule
    np.save(os.path.join(save_path, 'final_schedule.npy'), env.schedule_grid)


if __name__ == '__main__':
    train_model()
