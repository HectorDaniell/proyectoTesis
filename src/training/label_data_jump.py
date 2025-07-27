import pandas as pd

# Function to label performance based on jump height analysis
def label_performance_jump(csv_file):
    """
    Label exercise performance for jumping movements based on ankle height analysis.
    
    This function analyzes the vertical movement of ankles during jump exercises
    to determine performance quality. Higher jumps (lower ankle Y coordinates)
    indicate better performance.
    
    Args:
        csv_file (str): Path to CSV file containing pose landmark data
        
    Returns:
        str: Path to the labeled CSV file with performance classifications
    """
    df = pd.read_csv(csv_file)

    # Extract ankle landmark coordinates for height analysis
    right_ankle_y = df['right_ankle_y']
    left_ankle_y = df['left_ankle_y']

    # Calculate average ankle height (lower Y values = higher jump)
    df['avg_ankle_height'] = (right_ankle_y + left_ankle_y) / 2

    # Initialize performance column with default values
    df['performance'] = 0

    # Define performance thresholds based on ankle height percentiles
    # Lower Y values indicate higher jumps (better performance)
    high_threshold = df['avg_ankle_height'].quantile(0.33)  # Top 33% = High performance
    low_threshold = df['avg_ankle_height'].quantile(0.66)   # Bottom 33% = Low performance

    # Assign performance labels based on ankle height thresholds
    df.loc[df['avg_ankle_height'] <= high_threshold, 'performance'] = 1  # High performance
    df.loc[(df['avg_ankle_height'] > high_threshold) & (df['avg_ankle_height'] <= low_threshold), 'performance'] = 2  # Moderate performance
    df.loc[df['avg_ankle_height'] > low_threshold, 'performance'] = 3  # Low performance

    # Save labeled data to new CSV file
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Jump performance labels saved in {labeled_csv}")

    return labeled_csv
