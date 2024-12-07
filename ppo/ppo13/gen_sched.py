import numpy as np
import pandas as pd
import glob
import os
import re

def find_latest_schedule_file(schedule_dir='schedules'):
    """
    Finds the latest schedule numpy file in the specified directory based on the highest timestep number.
    
    Args:
        schedule_dir (str): Path to the directory containing schedule .npy files.
        
    Returns:
        str or None: Path to the latest schedule file or None if no files are found.
    """
    # Pattern to match files like 'schedule_step_10000.npy'
    pattern = os.path.join(schedule_dir, 'schedule_step_*.npy')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No schedule files found in directory '{schedule_dir}'.")
        return None
    
    # Extract timestep numbers from filenames
    timestep_files = {}
    for file in files:
        match = re.search(r'schedule_step_(\d+)\.npy', os.path.basename(file))
        if match:
            timestep = int(match.group(1))
            timestep_files[timestep] = file
    
    if not timestep_files:
        print(f"No schedule files with the correct naming pattern found in '{schedule_dir}'.")
        return None
    
    # Find the file with the highest timestep
    latest_timestep = max(timestep_files.keys())
    latest_file = timestep_files[latest_timestep]
    
    print(f"Latest schedule file found: '{latest_file}' (Timestep: {latest_timestep})")
    return latest_file

def load_schedule(file_path):
    """
    Loads the schedule grid from a numpy file.
    
    Args:
        file_path (str): Path to the .npy schedule file.
        
    Returns:
        np.ndarray: Schedule grid array.
    """
    try:
        schedule_grid = np.load(file_path)
        print(f"Schedule grid loaded successfully from '{file_path}'.")
        return schedule_grid
    except Exception as e:
        print(f"Error loading schedule file '{file_path}': {e}")
        return None

def schedule_grid_to_dataframe(schedule_grid):
    """
    Converts the schedule grid to a pandas DataFrame.
    
    Args:
        schedule_grid (np.ndarray): Schedule grid array with shape (num_days, time_slots_per_day, num_positions).
        
    Returns:
        pd.DataFrame: DataFrame containing the schedule.
    """
    num_days, time_slots_per_day, num_positions = schedule_grid.shape
    data = []
    
    for day in range(num_days):
        for time_slot in range(time_slots_per_day):
            for position in range(num_positions):
                controller = schedule_grid[day, time_slot, position]
                controller_assigned = controller if controller != -1 else 'Unassigned'
                data.append({
                    'Day': day + 1,  # Assuming Day 1 to Day 7
                    'Time Slot': time_slot + 1,  # Assuming Time Slot 1 to 24
                    'Position': position + 1,  # Assuming Position 1 to 10
                    'Controller Assigned': controller_assigned
                })
    
    df = pd.DataFrame(data)
    return df

def save_schedule_csv(df, output_path='schedules/latest_schedule.csv'):
    """
    Saves the schedule DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): Schedule DataFrame.
        output_path (str): Path to save the CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Schedule CSV saved to '{output_path}'.")

def compute_and_print_stats(schedule_grid):
    """
    Computes and prints statistics about unassigned slots.
    
    Args:
        schedule_grid (np.ndarray): Schedule grid array.
    """
    total_slots = schedule_grid.size
    unassigned_slots = np.sum(schedule_grid == -1)
    percentage_unassigned = (unassigned_slots / total_slots) * 100
    
    print("\n--- Schedule Statistics ---")
    print(f"Total Slots: {total_slots}")
    print(f"Unassigned Slots: {unassigned_slots}")
    print(f"Percentage Unassigned: {percentage_unassigned:.2f}%\n")
    
    # Unassigned slots per day
    num_days = schedule_grid.shape[0]
    unassigned_per_day = np.sum(schedule_grid == -1, axis=(1,2))
    for day, count in enumerate(unassigned_per_day, start=1):
        print(f"Day {day}: {count} unassigned slots.")
    
    print()
    
    # Unassigned slots per position
    num_positions = schedule_grid.shape[2]
    unassigned_per_position = np.sum(schedule_grid == -1, axis=(0,1))
    for position, count in enumerate(unassigned_per_position, start=1):
        print(f"Position {position}: {count} unassigned slots.")
    
    # Optionally, list all unassigned slots
    # Uncomment the following lines if you want to see all unassigned slots details
    """
    unassigned_indices = np.argwhere(schedule_grid == -1)
    if unassigned_indices.size > 0:
        print("\nList of Unassigned Slots:")
        for idx in unassigned_indices:
            day, time_slot, position = idx
            print(f"Day {day+1}, Time Slot {time_slot+1}, Position {position+1}")
    else:
        print("No unassigned slots found.")
    """

def main():
    # Define the schedule directory
    schedule_dir = 'schedules'
    
    # Find the latest schedule file
    latest_file = find_latest_schedule_file(schedule_dir)
    
    if latest_file is None:
        return  # Exit if no schedule file is found
    
    # Load the schedule grid
    schedule_grid = load_schedule(latest_file)
    
    if schedule_grid is None:
        return  # Exit if loading failed
    
    # Convert schedule grid to DataFrame
    df_schedule = schedule_grid_to_dataframe(schedule_grid)
    
    # Save the DataFrame to CSV
    output_csv_path = os.path.join(schedule_dir, 'latest_schedule.csv')
    save_schedule_csv(df_schedule, output_csv_path)
    
    # Compute and print statistics
    compute_and_print_stats(schedule_grid)

if __name__ == "__main__":
    main()
