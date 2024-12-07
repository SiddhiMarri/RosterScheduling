import numpy as np
import os
import csv

def find_latest_schedule_file(schedules_dir='schedules'):
    # Get list of .npy files in the schedules directory
    files = [f for f in os.listdir(schedules_dir) if f.endswith('.npy')]
    if not files:
        print("No schedule files found in the directory.")
        return None
    # Get the latest file based on modification time
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(schedules_dir, x)))
    return os.path.join(schedules_dir, latest_file)

def convert_schedule_to_csv_reorganized(schedule_grid, output_csv='schedule.csv'):
    num_days, time_slots_per_day, num_positions = schedule_grid.shape

    # Generate column headers
    headers = ['Position'] + [f'Day{day+1}_Slot{time_slot+1}' for day in range(num_days) for time_slot in range(time_slots_per_day)]

    # Prepare the data in the desired format
    data = []
    for position in range(num_positions):
        row = [position + 1]  # Start with the position number
        for day in range(num_days):
            for time_slot in range(time_slots_per_day):
                row.append(schedule_grid[day, time_slot, position])
        data.append(row)

    # Write to the CSV file
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write the header
        writer.writerows(data)    # Write the data rows

if __name__ == "__main__":
    schedules_dir = 'schedules'
    output_csv = 'schedule_parsed.csv'
    latest_schedule_file = find_latest_schedule_file(schedules_dir)
    if latest_schedule_file:
        print(f"Loading schedule from {latest_schedule_file}")
        schedule_grid = np.load(latest_schedule_file)
        convert_schedule_to_csv_reorganized(schedule_grid, output_csv)
        print(f"Schedule saved to {output_csv}")
    else:
        print("No schedule file found to convert.")
