import pandas as pd
import numpy as np

# Function to label performance based on crawling exercise analysis
def label_performance_crawl(csv_file):
    """
    Label exercise performance for crawling movements using multiple biomechanical metrics.
    
    This function evaluates crawling performance using three key metrics:
    1. Diagonal coordination (cross-limb movement patterns)
    2. Hip stability (maintaining consistent hip height)
    3. Movement fluidity (smooth vs. jerky movements)
    
    Args:
        csv_file (str): Path to CSV file containing pose landmark data
        
    Returns:
        str: Path to the labeled CSV file with performance classifications
    """
    df = pd.read_csv(csv_file)
    
    # 1. Calculate diagonal coordination (correlation between cross movements)
    # Extract wrist and knee movement patterns for coordination analysis
    right_wrist_x = df['right_wrist_x'].diff()  # Position change over time
    left_knee_x = df['left_knee_x'].diff()      # Position change over time
    left_wrist_x = df['left_wrist_x'].diff()    # Position change over time
    right_knee_x = df['right_knee_x'].diff()    # Position change over time
    
    # Calculate coordination between opposite limbs (should be negative in correct crawling)
    # Proper crawling involves opposite arm and leg moving together
    df['right_left_coordination'] = (right_wrist_x * left_knee_x).fillna(0)
    df['left_right_coordination'] = (left_wrist_x * right_knee_x).fillna(0)
    df['coordination_score'] = (df['right_left_coordination'] + df['left_right_coordination']).rolling(window=10).mean().fillna(0)
    
    # 2. Evaluate hip stability (variation in hip height during movement)
    hip_height = (df['right_hip_y'] + df['left_hip_y']) / 2
    # Lower variation indicates better stability and control
    df['hip_stability'] = 1 - hip_height.rolling(window=10).std().fillna(0)
    
    # 3. Calculate movement fluidity (smooth vs. abrupt velocity changes)
    hip_velocity = hip_height.diff().abs()
    # Smoother movements have lower standard deviation relative to mean velocity
    # Handle division by zero and NaN values
    velocity_std = hip_velocity.rolling(window=10).std().fillna(0)
    velocity_mean = hip_velocity.rolling(window=10).mean().fillna(0)
    # Avoid division by zero
    velocity_ratio = velocity_std / velocity_mean.replace(0, 1)
    df['movement_fluidity'] = (1 - velocity_ratio).fillna(0)
    
    # Combine all metrics into a comprehensive performance score
    # Weighted combination: 40% coordination, 30% stability, 30% fluidity
    df['total_score'] = (
        df['coordination_score'].abs().fillna(0) * 0.4 +  # 40% coordination
        df['hip_stability'].fillna(0) * 0.3 +             # 30% stability
        df['movement_fluidity'].fillna(0) * 0.3           # 30% fluidity
    )
    
    # Classify performance based on total score percentiles
    high_threshold = df['total_score'].quantile(0.67)  # Top 33%
    low_threshold = df['total_score'].quantile(0.33)   # Bottom 33%
    
    # Assign performance labels
    df['performance'] = 2  # Default value: moderate
    df.loc[df['total_score'] >= high_threshold, 'performance'] = 1  # High performance
    df.loc[df['total_score'] <= low_threshold, 'performance'] = 3   # Low performance
    
    # Save labeled results
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Crawling performance labels saved in {labeled_csv}")
    
    return labeled_csv
