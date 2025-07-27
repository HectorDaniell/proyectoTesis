import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe components for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Function to process a video and extract pose landmarks using MediaPipe
def process_video(video_path, exercise_name, combined_df):
    """
    Extract pose landmarks from a video file using MediaPipe Holistic model.
    
    This function processes a video frame by frame, detecting pose landmarks
    and storing their coordinates for further analysis.
    
    Args:
        video_path (str): Path to the input video file
        exercise_name (str): Name of the exercise for reference
        combined_df (pd.DataFrame): Existing DataFrame to append new data to
        
    Returns:
        pd.DataFrame: Combined DataFrame with landmark data from all processed videos
    """
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        landmark_data = []
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Extract pose landmarks if detected in the frame
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Store x, y, z coordinates for each landmark point
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmark_data.append(frame_landmarks)
        
        cap.release()

    # Define column names according to MediaPipe pose landmarks
    # These correspond to the 33 pose landmarks detected by MediaPipe
    body_parts = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
        'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 
        'left_foot_index', 'right_foot_index', 'left_pinky', 'right_pinky', 
        'left_index', 'right_index', 'left_thumb', 'right_thumb', 
        'left_foot', 'right_foot'
    ]

    # Generate column names for x, y, z coordinates of each body part
    column_names = []
    for part in body_parts:
        column_names.append(f"{part}_x")  # X coordinate
        column_names.append(f"{part}_y")  # Y coordinate 
        column_names.append(f"{part}_z")  # Z coordinate

    # Convert landmark data to DataFrame with descriptive column names
    df = pd.DataFrame(landmark_data, columns=column_names)
    
    # Concatenate with existing combined DataFrame to accumulate data from multiple videos
    return pd.concat([combined_df, df], ignore_index=True)
