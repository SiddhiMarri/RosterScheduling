import os
import numpy as np
import pandas as pd
from datetime import datetime

def find_latest_schedule(schedule_dir='schedules'):
    """
    Finds the latest .npy schedule file in the specified directory.
    
    Args:
        schedule_dir (str): Path to the schedules directory.
        
    Returns:
        str: Path to the latest schedule file.
    """
    # List all .npy files in the directory
    npy_files = [f for f in os.listdir(schedule_dir) if f.endswith('.npy')]
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in the directory '{schedule_dir}'.")
    
    # Sort files by modification time
    npy_files.sort(key=lambda x: os.path.getmtime(os.path.join(schedule_dir, x)), reverse=True)
    
    latest_file = npy_files[0]
    latest_path = os.path.join(schedule_dir, latest_file)
    
    print(f"Latest schedule file found: {latest_file}")
    return latest_path

def load_schedule(file_path):
    """
    Loads the schedule grid from a .npy file.
    
    Args:
        file_path (str): Path to the .npy schedule file.
        
    Returns:
        np.ndarray: Schedule grid.
    """
    schedule_grid = np.load(file_path)
    print(f"Schedule grid loaded with shape {schedule_grid.shape}.")
    return schedule_grid

def convert_schedule_to_csv(schedule_grid, output_csv='latest_schedule.csv', num_days=7, time_slots_per_day=24, num_positions=10):
    """
    Converts the schedule grid to a readable CSV format.
    
    Args:
        schedule_grid (np.ndarray): The schedule grid.
        output_csv (str): Path to save the CSV file.
        num_days (int): Number of days in the schedule.
        time_slots_per_day (int): Number of time slots per day.
        num_positions (int): Number of positions.
        
    Returns:
        pd.DataFrame: DataFrame containing the schedule.
    """
    total_time_slots, positions = schedule_grid.shape
    assert positions == num_positions, "Number of positions does not match."
    
    data = []
    for time_slot in range(total_time_slots):
        day = time_slot // time_slots_per_day + 1  # Days start at 1
        hour = time_slot % time_slots_per_day      # Hours 0-23
        for position in range(num_positions):
            controller_id = schedule_grid[time_slot, position]
            data.append({
                'Day': day,
                'Hour': hour,
                'Position': position + 1,  # Positions start at 1
                'Controller_ID': controller_id
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Schedule successfully saved to '{output_csv}'.")
    return df

def compute_detailed_stats(schedule_grid, num_days=7, time_slots_per_day=24, num_positions=10, num_controllers=100):
    """
    Computes and prints detailed statistics from the schedule grid.
    
    Args:
        schedule_grid (np.ndarray): The schedule grid.
        num_days (int): Number of days in the schedule.
        time_slots_per_day (int): Number of time slots per day.
        num_positions (int): Number of positions.
        num_controllers (int): Total number of controllers.
    """
    total_time_slots, positions = schedule_grid.shape
    assert positions == num_positions, "Number of positions does not match."
    
    # Initialize statistics dictionaries
    unassigned_slots = {day: {pos:0 for pos in range(1, num_positions+1)} for day in range(1, num_days+1)}
    controller_assignments = {c:0 for c in range(num_controllers)}
    assignments_per_day = {day:0 for day in range(1, num_days+1)}
    assignments_per_position = {pos:0 for pos in range(1, num_positions+1)}
    
    unique_controllers = set()
    
    for time_slot in range(total_time_slots):
        day = time_slot // time_slots_per_day + 1
        for position in range(1, num_positions+1):
            controller_id = schedule_grid[time_slot, position-1]
            if controller_id == -1:
                unassigned_slots[day][position] += 1
            else:
                if 0 <= controller_id < num_controllers:
                    controller_assignments[controller_id] += 1
                    unique_controllers.add(controller_id)
                    assignments_per_day[day] += 1
                    assignments_per_position[position] += 1
                else:
                    print(f"Warning: Controller ID {controller_id} out of range.")
    
    # Total Unassigned Slots per Day per Position
    print("\n=== Total Unassigned Slots per Day per Position ===")
    unassigned_df = pd.DataFrame(unassigned_slots).T
    unassigned_df.index.name = 'Day'
    unassigned_df.columns = [f'Position_{p}' for p in unassigned_df.columns]
    print(unassigned_df)
    
    # Total Assignments per Controller
    print("\n=== Total Assignments per Controller ===")
    assignments_df = pd.DataFrame(list(controller_assignments.items()), columns=['Controller_ID', 'Total_Assignments'])
    assignments_df = assignments_df.sort_values(by='Total_Assignments', ascending=False)
    print(assignments_df)
    
    # Controller Utilization
    print("\n=== Controller Utilization (%) ===")
    controller_utilization = {c: (count / total_time_slots) * 100 for c, count in controller_assignments.items()}
    utilization_df = pd.DataFrame(list(controller_utilization.items()), columns=['Controller_ID', 'Utilization (%)'])
    utilization_df = utilization_df.sort_values(by='Utilization (%)', ascending=False)
    print(utilization_df)
    
    # Overall Schedule Completeness
    print("\n=== Overall Schedule Completeness ===")
    total_positions = total_time_slots * num_positions
    filled_positions = total_positions - np.sum(schedule_grid == -1)
    completeness = (filled_positions / total_positions) * 100
    print(f"Overall Schedule Completeness: {completeness:.2f}%")
    
    # Total Unique Controllers Used
    print("\n=== Total Unique Controllers Used ===")
    total_unique = len(unique_controllers)
    print(f"Total Unique Controllers Assigned: {total_unique} out of {num_controllers}")
    
    # Assignments Distribution
    print("\n=== Assignments Distribution ===")
    distribution = assignments_df['Total_Assignments'].value_counts().sort_index()
    distribution_df = pd.DataFrame({
        'Number_of_Assignments': distribution.index,
        'Number_of_Controllers': distribution.values
    })
    print(distribution_df)
    
    # Top 5 Controllers with Most Assignments
    print("\n=== Top 5 Controllers with Most Assignments ===")
    top_5 = assignments_df.head(5)
    print(top_5)
    
    # Bottom 5 Controllers with Least Assignments
    print("\n=== Bottom 5 Controllers with Least Assignments ===")
    bottom_5 = assignments_df.tail(5)
    print(bottom_5)
    
    # Assignments per Day
    print("\n=== Assignments per Day ===")
    assignments_day_df = pd.DataFrame(list(assignments_per_day.items()), columns=['Day', 'Total_Assignments'])
    print(assignments_day_df)
    
    # Assignments per Position
    print("\n=== Assignments per Position ===")
    assignments_position_df = pd.DataFrame(list(assignments_per_position.items()), columns=['Position', 'Total_Assignments'])
    print(assignments_position_df)
    
    # Average Assignments per Controller
    print("\n=== Average Assignments per Controller ===")
    average_assignments = assignments_df['Total_Assignments'].mean()
    print(f"Average Assignments per Controller: {average_assignments:.2f}")
    
    # Standard Deviation of Assignments
    print("\n=== Standard Deviation of Assignments ===")
    std_assignments = assignments_df['Total_Assignments'].std()
    print(f"Standard Deviation of Assignments: {std_assignments:.2f}")
    
    # Additional Insights
    print("\n=== Additional Insights ===")
    
    # Convert controller_assignments to Pandas Series for easier computation
    assignments_series = pd.Series(controller_assignments)
    
    # Percentage of Controllers with No Assignments
    no_assignment_count = (assignments_series == 0).sum()
    no_assignment_percentage = (no_assignment_count / num_controllers) * 100
    print(f"Percentage of Controllers with No Assignments: {no_assignment_percentage:.2f}%")
    
    # Percentage of Controllers with Assignments >= 50
    high_assignment_count = (assignments_series >= 50).sum()
    high_assignment_percentage = (high_assignment_count / num_controllers) * 100
    print(f"Percentage of Controllers with Assignments >= 50: {high_assignment_percentage:.2f}%")
    
    # Potential Visualization Suggestions
    print("\n=== Visualization Suggestions ===")
    print("- Heatmap of Assignments per Controller per Day")
    print("- Bar Chart of Assignments Distribution")
    print("- Pie Chart of Controller Utilization")
    print("- Line Graph of Assignments Over Days")

def main():
    # Directory where schedules are saved
    schedule_dir = 'schedules'
    
    try:
        # Step 1: Find the latest schedule file
        latest_schedule_path = find_latest_schedule(schedule_dir)
        
        # Step 2: Load the schedule grid
        schedule_grid = load_schedule(latest_schedule_path)
        
        # Step 3: Convert schedule to CSV
        output_csv = 'latest_schedule.csv'
        schedule_df = convert_schedule_to_csv(schedule_grid, output_csv=output_csv)
        
        # Step 4: Compute and print detailed statistics
        compute_detailed_stats(schedule_grid)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
