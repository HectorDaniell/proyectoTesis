import pandas as pd
import numpy as np

# Function to label performance based on sitting exercise analysis
def label_performance_sit(csv_file):
    """
    Label exercise performance for sitting/standing movements using postural analysis.
    
    This function evaluates sitting performance using three key metrics:
    1. Postural control (spine alignment with vertical)
    2. Body symmetry (balance between left and right sides)
    3. Transition smoothness (quality of sit-to-stand movements)
    
    Args:
        csv_file (str): Path to CSV file containing pose landmark data
        
    Returns:
        str: Path to the labeled CSV file with performance classifications
    """
    df = pd.read_csv(csv_file)
    
    # 1. Evaluate postural control (spine-head alignment)
    # Calculate spine vector from hips to shoulders
    spine_vector_x = df['left_shoulder_x'] + df['right_shoulder_x'] - df['left_hip_x'] - df['right_hip_x']
    spine_vector_y = df['left_shoulder_y'] + df['right_shoulder_y'] - df['left_hip_y'] - df['right_hip_y']
    
    # Calculate angle with respect to vertical (should be close to 0° in optimal sitting)
    # Perfect posture would have spine aligned with vertical axis
    df['spine_angle'] = np.abs(np.arctan2(spine_vector_x, spine_vector_y))
    df['posture_score'] = 1 - df['spine_angle'] / np.pi  # Normalized 0-1 (1 = vertical)
    
    # 2. Evaluate symmetry (difference between left and right sides)
    # Good posture should maintain balance between left and right body sides
    hip_symmetry = 1 - np.abs(df['left_hip_y'] - df['right_hip_y'])
    shoulder_symmetry = 1 - np.abs(df['left_shoulder_y'] - df['right_shoulder_y'])
    df['symmetry_score'] = (hip_symmetry + shoulder_symmetry) / 2
    
    # 3. Detect transitions (sit/stand) to evaluate smoothness
    hip_height = (df['left_hip_y'] + df['right_hip_y']) / 2
    hip_velocity = hip_height.diff().rolling(window=5).mean()
    
    # Identify transition periods (sustained changes in hip height)
    # These indicate sit-to-stand or stand-to-sit movements
    df['is_transition'] = (hip_velocity.abs() > hip_velocity.abs().quantile(0.7)).astype(int)
    
    # Calculate smoothness in transitions (lower acceleration = smoother movement)
    # Smooth transitions indicate better motor control
    df['transition_smoothness'] = 1 - hip_velocity.diff().abs() / hip_velocity.abs().max()
    
    # Combine all metrics into a comprehensive performance score
    # Weighted combination: 40% posture, 30% symmetry, 30% transition smoothness
    df['total_score'] = (
        df['posture_score'].fillna(0) * 0.4 +            # 40% posture
        df['symmetry_score'].fillna(0) * 0.3 +           # 30% symmetry
        df['transition_smoothness'].fillna(0) * 0.3      # 30% transition smoothness
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
    print(f"Sitting performance labels saved in {labeled_csv}")
    
    return labeled_csv
