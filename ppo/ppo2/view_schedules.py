import numpy as np
import os
import csv
import re

def load_schedule(file_path):
    """
    Loads a schedule from a .npy file.

    Parameters:
    - file_path (str): Path to the .npy file.

    Returns:
    - np.ndarray: The loaded schedule array or None if file doesn't exist.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    try:
        schedule = np.load(file_path)
        return schedule
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None

def print_schedule(schedule):
    """
    Prints the schedule in a readable format.

    Parameters:
    - schedule (np.ndarray): The schedule array with shape (num_days, time_slots_per_day, num_positions).
    """
    num_days, time_slots_per_day, num_positions = schedule.shape
    total_unassigned = 0  # Initialize counter for unassigned slots

    for day in range(num_days):
        print(f"\n=== Day {day + 1} ===")
        for slot in range(time_slots_per_day):
            print(f"  Time Slot {slot + 1}:")
            for position in range(num_positions):
                controller = schedule[day, slot, position]
                if controller == -1:
                    assignment = "Unassigned"
                    total_unassigned += 1  # Increment counter
                else:
                    assignment = f"Controller {controller}"
                print(f"    Position {position + 1}: {assignment}")

    print(f"\nTotal Unassigned Slots: {total_unassigned}")

def save_schedule_to_csv(schedule, csv_file_path):
    """
    Saves the schedule to a CSV file.

    Parameters:
    - schedule (np.ndarray): The schedule array.
    - csv_file_path (str): Path to save the CSV file.
    """
    num_days, time_slots_per_day, num_positions = schedule.shape
    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header
            header = ['Day', 'Time Slot'] + [f'Position {p+1}' for p in range(num_positions)]
            writer.writerow(header)
            # Write schedule data
            for day in range(num_days):
                for slot in range(time_slots_per_day):
                    row = [day + 1, slot + 1]
                    for position in range(num_positions):
                        controller = schedule[day, slot, position]
                        assignment = f"Controller {controller}" if controller != -1 else "Unassigned"
                        row.append(assignment)
                    writer.writerow(row)
        print(f"\nSchedule successfully saved to '{csv_file_path}'.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def get_latest_schedule_file(schedules_dir):
    """
    Retrieves the latest schedule file based on the highest step number.

    Parameters:
    - schedules_dir (str): Directory where schedule files are stored.

    Returns:
    - str or None: The latest schedule file name or None if no files are found.
    """
    if not os.path.isdir(schedules_dir):
        print(f"Error: Directory '{schedules_dir}' does not exist.")
        return None
    schedule_files = [f for f in os.listdir(schedules_dir) if f.endswith('.npy')]
    if not schedule_files:
        print(f"No .npy schedule files found in '{schedules_dir}'.")
        return None

    # Use regex to extract step numbers
    pattern = re.compile(r'schedule_step_(\d+)\.npy')
    schedule_steps = {}
    for file in schedule_files:
        match = pattern.match(file)
        if match:
            step = int(match.group(1))
            schedule_steps[file] = step

    if not schedule_steps:
        print(f"No schedule files matching the pattern 'schedule_step_{{step}}.npy' found in '{schedules_dir}'.")
        return None

    # Find the file with the highest step number
    latest_file = max(schedule_steps, key=schedule_steps.get)
    return latest_file

def main():
    """
    Main function to view the latest schedule.
    """
    print("=== ATC Latest Schedule Viewer ===")
    schedules_dir = 'schedules'  # Directory where schedule .npy files are stored

    # Get the latest schedule file
    latest_file = get_latest_schedule_file(schedules_dir)
    if not latest_file:
        return

    file_path = os.path.join(schedules_dir, latest_file)
    print(f"\nLatest schedule file found: '{latest_file}'")

    schedule = load_schedule(file_path)
    if schedule is None:
        return

    print(f"\n=== Displaying Schedule from '{latest_file}' ===")
    print_schedule(schedule)

    # Ask user if they want to save the schedule to CSV
    save_csv = input("\nWould you like to save this schedule to a CSV file? (y/n): ").strip().lower()
    if save_csv == 'y':
        csv_file_name = os.path.splitext(latest_file)[0] + '.csv'
        csv_file_path = os.path.join(schedules_dir, csv_file_name)
        save_schedule_to_csv(schedule, csv_file_path)
    else:
        print("CSV export skipped.")

if __name__ == "__main__":
    main()
