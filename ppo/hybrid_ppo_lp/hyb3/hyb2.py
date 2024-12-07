# Import necessary libraries
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import pulp

# Define the scheduling environment


class ATCSchedulingEnv(gym.Env):
    def __init__(self):
        super(ATCSchedulingEnv, self).__init__()

        # Define constants
        self.num_controllers = 100
        self.num_positions = 10
        self.num_days = 7
        self.time_slots_per_day = 24  # Each time slot represents 1 hour
        self.total_time_slots = self.num_days * self.time_slots_per_day

        # Action space: Priority scores for each controller
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_controllers,),
            dtype=np.float32
        )

        # Observation space: Controller states
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_controllers, 8),  # 8 state features per controller
            dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def reset(self):
        # Initialize controller states
        self.controller_states = np.zeros(
            (self.num_controllers, 8), dtype=np.float32)
        # State features:
        # 0: Total Hours Worked Today
        # 1: Total Hours Worked This Week
        # 2: Total Hours Worked This Month
        # 3: Consecutive Duty Days
        # 4: Remaining Break Time (hours)
        # 5: Worked Today Flag (0 or 1)
        # 6: Fatigue Level
        # 7: Worked Yesterday Flag (0 or 1)

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
        # Use LP optimization to get assignments
        assignments = optimize_schedule(action, self.controller_states, self)

        # Update controller states
        self._update_controller_states(assignments)

        # Save assignments to schedule grid
        self.schedule_grid[self.current_time_slot] = assignments

        # Compute reward
        reward = self._compute_reward(assignments)

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

    def _update_controller_states(self, assignments):
        # Decrease remaining break time and ensure it's not negative
        for c in range(self.num_controllers):
            if self.controller_states[c][4] > 0:
                self.controller_states[c][4] -= 1  # Decrease by 1 hour
                self.controller_states[c][4] = max(
                    self.controller_states[c][4], 0)

        for controller_id in assignments:
            if controller_id != -1:
                # Update working hours
                self.controller_states[controller_id][0] += 1  # Today
                self.controller_states[controller_id][1] += 1  # This Week
                self.controller_states[controller_id][2] += 1  # This Month

                # Update consecutive duty days
                if self.controller_states[controller_id][7] == 1:
                    # Worked yesterday, increment consecutive duty days
                    self.controller_states[controller_id][3] += 1
                else:
                    # Did not work yesterday, reset consecutive duty days to 1
                    self.controller_states[controller_id][3] = 1

                # Set worked today flag
                self.controller_states[controller_id][5] = 1

                # Enforce mandatory breaks after 2 hours of continuous work
                if self.controller_states[controller_id][0] % 2 == 0:
                    # Break time in hours
                    self.controller_states[controller_id][4] = 0.5

                # Increase fatigue level (optional, can be adjusted)
                self.controller_states[controller_id][6] += 0.01

        # Normalize fatigue level to be between 0 and 1
        self.controller_states[:, 6] = np.clip(
            self.controller_states[:, 6], 0, 1)

    def _reset_daily_counters(self):
        for controller_id in range(self.num_controllers):
            # Set worked yesterday flag based on whether they worked today
            self.controller_states[controller_id][7] = self.controller_states[controller_id][5]

            # Reset worked today flag
            self.controller_states[controller_id][5] = 0

            # Reset daily hours
            self.controller_states[controller_id][0] = 0  # Today

            # Check for 48-hour break after 6 consecutive days
            if self.controller_states[controller_id][3] >= 6:
                # Set break time in hours
                # Break Time in hours
                self.controller_states[controller_id][4] = 48
                self.controller_states[controller_id][3] = 0  # Reset Duty Days

    def _compute_reward(self, assignments):
        # Reward for filling all positions
        if -1 not in assignments:
            reward = 100  # Large reward for fully staffed positions
        else:
            unfilled_positions = assignments.count(-1)
            reward = -100 * unfilled_positions  # Heavy penalty for unfilled positions

        # Penalty for using more controllers (to minimize the number used)
        num_controllers_used = len(set([a for a in assignments if a != -1]))
        reward -= num_controllers_used * 0.1

        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Define the optimization function


def optimize_schedule(controller_priorities, controller_states, env):
    # Create the LP problem
    prob = pulp.LpProblem("ATC_Scheduling", pulp.LpMinimize)

    # Decision variables
    x = {}
    for c in range(env.num_controllers):
        for p in range(env.num_positions):
            var_name = f'x_{c}_{p}'
            x[(c, p)] = pulp.LpVariable(var_name, cat='Binary')

    # Objective: Minimize weighted assignments
    # Weights combine priorities and total hours worked to balance workload
    alpha = 1.0  # Weight for balancing working hours
    total_hours_worked_today = controller_states[:, 0]  # Today

    weights = {}
    for c in range(env.num_controllers):
        weights[c] = controller_priorities[c] + \
            alpha * total_hours_worked_today[c]

    prob += pulp.lpSum(weights[c] * x[(c, p)] for c in range(env.num_controllers)
                       for p in range(env.num_positions))

    # Constraints:
    # Each position must be filled
    for p in range(env.num_positions):
        prob += pulp.lpSum(x[(c, p)] for c in range(env.num_controllers)) == 1

    # Controllers can work at most one position per time slot
    for c in range(env.num_controllers):
        prob += pulp.lpSum(x[(c, p)] for p in range(env.num_positions)) <= 1

    # Controller availability constraints
    for c in range(env.num_controllers):
        controller_state = controller_states[c]

        # Max Consecutive Duty Days
        if controller_state[3] >= 6:
            for p in range(env.num_positions):
                prob += x[(c, p)] == 0

        # Break Time
        if controller_state[4] > 0:
            for p in range(env.num_positions):
                prob += x[(c, p)] == 0

        # Fatigue Level Threshold (example constraint)
        if controller_state[6] >= 1.0:
            for p in range(env.num_positions):
                prob += x[(c, p)] == 0

        # Max Daily Duty Hours (assume max 12 hours per day)
        if controller_state[0] >= 12:
            for p in range(env.num_positions):
                prob += x[(c, p)] == 0

    # Solve the problem
    prob.solve()

    # Extract the assignments
    optimized_assignments = [-1] * env.num_positions
    for p in range(env.num_positions):
        for c in range(env.num_controllers):
            if pulp.value(x[(c, p)]) == 1:
                optimized_assignments[p] = c

    return optimized_assignments

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

# Main training loop


def train_model():
    env = ATCSchedulingEnv()

    # Create save directory
    save_path = './schedules'
    os.makedirs(save_path, exist_ok=True)

    # Create the RL model
    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log="./ppo_tensorboard/")

    # Create the callback
    callback = SaveScheduleCallback(
        env, check_freq=10000, save_path=save_path)

    # Train the model
    model.learn(total_timesteps=100000000, callback=callback)

    # Save the final schedule
    np.save(os.path.join(save_path, 'final_schedule.npy'), env.schedule_grid)


if __name__ == '__main__':
    train_model()
