import pandas as pd
import numpy as np

# Function to label performance based on ball throwing exercise analysis
def label_performance_ball_throwing(csv_file):
    """
    Label exercise performance for ball throwing movements using kinematic analysis.
    
    This function evaluates throwing performance using three key metrics:
    1. Kinematic sequence (proper proximal-to-distal joint activation)
    2. Movement amplitude (range of motion during throw)
    3. Postural stability (maintaining balance during throwing motion)
    
    Args:
        csv_file (str): Path to CSV file containing pose landmark data
        
    Returns:
        str: Path to the labeled CSV file with performance classifications
    """
    df = pd.read_csv(csv_file)
    
    # 1. Evaluate kinematic sequence (correct order of joint activation)
    # Calculate movement velocities for each joint in the throwing chain
    trunk_rotation = np.arctan2(df['right_shoulder_x'] - df['left_shoulder_x'], 
                               df['right_shoulder_y'] - df['left_shoulder_y']).diff()
    shoulder_movement = (df['right_shoulder_x'].diff()**2 + df['right_shoulder_y'].diff()**2) ** 0.5
    elbow_movement = (df['right_elbow_x'].diff()**2 + df['right_elbow_y'].diff()**2) ** 0.5
    wrist_movement = (df['right_wrist_x'].diff()**2 + df['right_wrist_y'].diff()**2) ** 0.5
    
    # Smooth signals to reduce noise and improve peak detection
    window = 5
    trunk_rotation_smooth = trunk_rotation.abs().rolling(window=window).mean().fillna(0)
    shoulder_movement_smooth = shoulder_movement.rolling(window=window).mean().fillna(0)
    elbow_movement_smooth = elbow_movement.rolling(window=window).mean().fillna(0)
    wrist_movement_smooth = wrist_movement.rolling(window=window).mean().fillna(0)
    
    # Find activity peaks for each joint using 80th percentile threshold
    threshold_trunk = trunk_rotation_smooth.quantile(0.8)
    threshold_shoulder = shoulder_movement_smooth.quantile(0.8)
    threshold_elbow = elbow_movement_smooth.quantile(0.8)
    threshold_wrist = wrist_movement_smooth.quantile(0.8)
    
    # Create binary activity indicators for each joint
    df['trunk_active'] = (trunk_rotation_smooth > threshold_trunk).astype(int)
    df['shoulder_active'] = (shoulder_movement_smooth > threshold_shoulder).astype(int)
    df['elbow_active'] = (elbow_movement_smooth > threshold_elbow).astype(int)
    df['wrist_active'] = (wrist_movement_smooth > threshold_wrist).astype(int)
    
    # Evaluate proximal to distal sequence (trunk → shoulder → elbow → wrist)
    # Proper throwing mechanics follow this sequence
    df['sequencing_score'] = 0.0
    
    # For each frame window, verify if sequence is correct
    for i in range(len(df) - window):
        seq_window = df.iloc[i:i+window]
        # Check if all joints are active during the movement window
        if (seq_window['trunk_active'].sum() > 0 and seq_window['shoulder_active'].sum() > 0 and
            seq_window['elbow_active'].sum() > 0 and seq_window['wrist_active'].sum() > 0):
            
            # Find first activation frame for each joint
            first_trunk = seq_window['trunk_active'].idxmax()
            first_shoulder = seq_window['shoulder_active'].idxmax()
            first_elbow = seq_window['elbow_active'].idxmax()
            first_wrist = seq_window['wrist_active'].idxmax()
            
            # Verify if sequence follows proper proximal-to-distal pattern
            # Ideal sequence: trunk < shoulder < elbow < wrist
            correct_sequence = (first_trunk <= first_shoulder and 
                                first_shoulder <= first_elbow and 
                                first_elbow <= first_wrist)
            
            # Assign score based on sequence correctness
            df.loc[i:i+window, 'sequencing_score'] = 1.0 if correct_sequence else 0.5
    
    # 2. Movement amplitude (maximum wrist distance from rest position)
    # Calculate rest position as average of first 10 frames
    rest_wrist_x = df['right_wrist_x'][:10].mean()
    rest_wrist_y = df['right_wrist_y'][:10].mean()
    
    # Calculate maximum displacement from rest position
    # Larger displacement indicates better range of motion
    df['wrist_displacement'] = ((df['right_wrist_x'] - rest_wrist_x)**2 + 
                               (df['right_wrist_y'] - rest_wrist_y)**2) ** 0.5
    max_displacement = df['wrist_displacement'].max()
    df['amplitude_score'] = df['wrist_displacement'] / max_displacement if max_displacement > 0 else 0
    
    # 3. Postural stability during throwing motion
    # Calculate hip movement to assess stability
    hip_movement = ((df['right_hip_x'].diff()**2 + df['right_hip_y'].diff()**2) ** 0.5).rolling(window=5).mean()
    hip_movement_max = hip_movement.max()
    # Lower hip movement indicates better stability
    df['stability_score'] = 1 - hip_movement / hip_movement_max if hip_movement_max > 0 else 1
    
    # Combine all metrics into a comprehensive performance score
    # Weighted combination: 40% sequence, 30% amplitude, 30% stability
    df['total_score'] = (
        df['sequencing_score'].fillna(0) * 0.4 +  # 40% movement sequence
        df['amplitude_score'].fillna(0) * 0.3 +   # 30% amplitude
        df['stability_score'].fillna(1) * 0.3     # 30% stability
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
    print(f"Ball throwing performance labels saved in {labeled_csv}")
    
    return labeled_csv
