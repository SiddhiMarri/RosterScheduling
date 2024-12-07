import numpy as np
import pandas as pd
import glob
import os
import re

def list_schedule_files(schedule_dir='schedules'):
    """
    Lists all schedule numpy files in the specified directory and gathers their statistics.

    Args:
        schedule_dir (str): Path to the directory containing schedule .npy files.

    Returns:
        list of dict: Each dictionary contains file path, filename, percentage unassigned, and number of controllers used.
    """
    pattern = os.path.join(schedule_dir, 'schedule_step_*.npy')
    files = glob.glob(pattern)

    if not files:
        print(f"No schedule files found in directory '{schedule_dir}'.")
        return []

    schedule_info_list = []

    for file in files:
        try:
            schedule_grid = np.load(file)
        except Exception as e:
            print(f"Error loading file '{file}': {e}")
            continue

        total_slots = schedule_grid.size
        unassigned_slots = np.sum(schedule_grid == -1)
        percentage_unassigned = (unassigned_slots / total_slots) * 100

        unique_controllers = np.unique(schedule_grid[schedule_grid != -1])
        number_of_controllers = len(unique_controllers)

        schedule_info = {
            'file_path': file,
            'filename': os.path.basename(file),
            'percentage_unassigned': percentage_unassigned,
            'number_of_controllers_used': number_of_controllers
        }

        schedule_info_list.append(schedule_info)

    return schedule_info_list

def display_schedule_summary(schedule_info_list):
    """
    Displays a summary table of all schedule files with their statistics, sorted by percentage unassigned in ascending order.

    Args:
        schedule_info_list (list of dict): List containing schedule file information.
    """
    if not schedule_info_list:
        print("No schedule files to display.")
        return

    # Sort the list by 'percentage_unassigned' in ascending order
    sorted_schedule = sorted(schedule_info_list, key=lambda x: x['percentage_unassigned'])

    print("\n--- Schedule Files Summary (Sorted by % Unassigned Ascending) ---\n")
    print(f"{'Index':<6} {'Filename':<30} {'% Unassigned':<15} {'# Controllers Used'}")
    print("-" * 70)
    for idx, info in enumerate(sorted_schedule, start=1):
        print(f"{idx:<6} {info['filename']:<30} {info['percentage_unassigned']:<15.2f} {info['number_of_controllers_used']}")
    print()

def prompt_user_selection(schedule_info_list):
    """
    Prompts the user to select a schedule file by index.

    Args:
        schedule_info_list (list of dict): List containing schedule file information.

    Returns:
        dict or None: The selected schedule file information or None if selection is invalid.
    """
    if not schedule_info_list:
        return None

    while True:
        try:
            selection = input(f"Enter the index of the schedule file to generate CSV (1-{len(schedule_info_list)}), or 'q' to quit: ").strip()
            if selection.lower() == 'q':
                return None
            index = int(selection)
            if 1 <= index <= len(schedule_info_list):
                return schedule_info_list[index - 1]
            else:
                print(f"Please enter a number between 1 and {len(schedule_info_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

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
                    'Day': day + 1,  # Assuming Day 1 to Day N
                    'Time Slot': time_slot + 1,  # Assuming Time Slot 1 to N
                    'Position': position + 1,  # Assuming Position 1 to N
                    'Controller Assigned': controller_assigned
                })

    df = pd.DataFrame(data)
    return df

def save_schedule_csv(df, output_path):
    """
    Saves the schedule DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Schedule DataFrame.
        output_path (str): Path to save the CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nSchedule CSV saved to '{output_path}'.\n")

def compute_and_print_stats(schedule_grid):
    """
    Computes and prints statistics about unassigned slots and controller usage.

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
    unassigned_per_day = np.sum(schedule_grid == -1, axis=(1, 2))
    for day, count in enumerate(unassigned_per_day, start=1):
        print(f"Day {day}: {count} unassigned slots.")

    print()

    # Unassigned slots per position
    num_positions = schedule_grid.shape[2]
    unassigned_per_position = np.sum(schedule_grid == -1, axis=(0, 1))
    for position, count in enumerate(unassigned_per_position, start=1):
        print(f"Position {position}: {count} unassigned slots.")

    print()

    # Total unique controllers used (excluding unassigned)
    unique_controllers = np.unique(schedule_grid[schedule_grid != -1])
    total_unique_controllers = len(unique_controllers)
    print(f"Total Unique Controllers Used: {total_unique_controllers}")
    print(f"Controller IDs: {unique_controllers.tolist()}\n")

    # Controller assignment distribution
    controller_ids, counts = np.unique(
        schedule_grid[schedule_grid != -1], return_counts=True)
    controller_distribution = dict(zip(controller_ids, counts))
    print("--- Controller Assignment Distribution ---")
    for controller, count in controller_distribution.items():
        print(f"Controller {controller}: {count} assigned slots.")

    print()

    # Most and Least Assigned Controllers
    if controller_distribution:
        max_assignments = max(controller_distribution.values())
        min_assignments = min(controller_distribution.values())
        most_assigned = [
            ctrl for ctrl, cnt in controller_distribution.items() if cnt == max_assignments]
        least_assigned = [
            ctrl for ctrl, cnt in controller_distribution.items() if cnt == min_assignments]

        print(
            f"Most Assigned Controller(s) ({max_assignments} slots): {most_assigned}")
        print(
            f"Least Assigned Controller(s) ({min_assignments} slots): {least_assigned}\n")
    else:
        print("No controllers assigned.\n")

def main():
    # Define the schedule directory
    schedule_dir = 'schedules'

    # List all schedule files with their statistics
    schedule_info_list = list_schedule_files(schedule_dir)

    if not schedule_info_list:
        return  # Exit if no schedule files are found

    # Display the summary of all schedule files sorted by percentage unassigned
    display_schedule_summary(schedule_info_list)

    # Prompt user to select a file to generate CSV
    selected_schedule = prompt_user_selection(schedule_info_list)

    if not selected_schedule:
        print("No file selected. Exiting.")
        return

    # Load the selected schedule grid
    schedule_grid = load_schedule(selected_schedule['file_path'])

    if schedule_grid is None:
        return  # Exit if loading failed

    # Convert schedule grid to DataFrame
    df_schedule = schedule_grid_to_dataframe(schedule_grid)

    # Define output CSV path
    output_csv_filename = os.path.splitext(selected_schedule['filename'])[0] + '.csv'
    output_csv_path = os.path.join(schedule_dir, output_csv_filename)

    # Save the DataFrame to CSV
    save_schedule_csv(df_schedule, output_csv_path)

    # Compute and print statistics
    compute_and_print_stats(schedule_grid)

def load_schedule(file_path):
    """
    Loads the schedule grid from a numpy file.

    Args:
        file_path (str): Path to the .npy schedule file.

    Returns:
        np.ndarray or None: Schedule grid array or None if loading fails.
    """
    try:
        schedule_grid = np.load(file_path)
        print(f"\nSchedule grid loaded successfully from '{file_path}'.")
        return schedule_grid
    except Exception as e:
        print(f"Error loading schedule file '{file_path}': {e}")
        return None

if __name__ == "__main__":
    main()
