import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# Initialize MediaPipe components for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def normalize_landmarks(frame_landmarks):
    """
    Apply body-reference normalization to a single frame's landmarks.

    Two-stage normalization:
    1. Translation: all points are translated so the hip center becomes the origin,
       ensuring invariance to the subject's position in the image.
    2. Scaling: all coordinates are divided by the shoulder distance, removing
       variability due to body size or camera distance.

    Args:
        frame_landmarks (list): Flat list of 99 values (33 landmarks x 3 coords)

    Returns:
        list: Normalized flat list of 99 values
    """
    coords = np.array(frame_landmarks).reshape(33, 3)

    # Stage 1: Translate using hip center as origin
    left_hip = coords[15]
    right_hip = coords[16]
    hip_center = (left_hip + right_hip) / 2
    coords = coords - hip_center

    # Stage 2: Scale by shoulder distance
    left_shoulder = coords[9]
    right_shoulder = coords[10]
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_distance > 0:
        coords = coords / shoulder_distance

    return coords.flatten().tolist()


# Function to process video and extract key points using MediaPipe pose detection
def process_new_video(video_path):
    """
    Extract pose landmarks from a video file using MediaPipe Holistic model.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        pd.DataFrame: DataFrame containing landmark coordinates for each frame
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
            
            # Extract pose landmarks if detected
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Store x, y, z coordinates for each landmark
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                # Apply body-reference normalization before storing
                landmark_data.append(normalize_landmarks(frame_landmarks))
        
        cap.release()

    # Convert landmark data to DataFrame for further processing
    df = pd.DataFrame(landmark_data)
    return df


# Function to calculate average performance across all frames
def calculate_average_performance(predictions):
    """
    Calculate overall performance score based on frame-by-frame predictions.
    
    Args:
        predictions (list): List of performance predictions (1=High, 2=Moderate, 3=Low)
        
    Returns:
        None: Prints performance summary to console
    """
    # Assign numeric values to performances: High=1, Moderate=2, Low=3
    performance_numeric = [1 if p == 1 else 2 if p == 2 else 3 for p in predictions]
    
    # Calculate average performance score
    average = sum(performance_numeric) / len(performance_numeric)
    
    # Determine overall performance category based on average score
    if average <= 1.5:
        overall_performance = "High"
    elif average <= 2.5:
        overall_performance = "Moderate"
    else:
        overall_performance = "Low"
    
    print(f"\nAverage performance: {average:.2f}")
    print(f"Overall exercise performance: {overall_performance}")


# Main function to load model and make predictions on a new video
def predict_performance(video_path, model_path, pca_path):
    """
    Complete pipeline for predicting exercise performance on a new video.
    
    Args:
        video_path (str): Path to input video file
        model_path (str): Path to trained model file (.pkl)
        pca_path (str): Path to the fitted PCA object (.pkl) saved during training
    """
    # Step 1: Process the new video to extract pose landmarks
    print("Processing video and extracting pose landmarks...")
    new_data_df = process_new_video(video_path)

    # Step 2: Apply the same PCA transformation used during training
    print("Loading PCA from training and applying transformation...")
    pca = joblib.load(pca_path)
    new_data_reduced = pca.transform(new_data_df)

    # Step 3: Load the trained model from disk
    print("Loading trained model...")
    model = joblib.load(model_path)
    
    # Step 4: Make predictions on the processed data
    print("Making predictions...")
    predictions = model.predict(new_data_reduced)

    # Step 5: Display frame-by-frame results
    print("\nFrame-by-frame performance analysis:")
    for i, pred in enumerate(predictions):
        if pred == 1:
            print(f"Frame {i}: High Performance")
        elif pred == 2:
            print(f"Frame {i}: Moderate Performance")
        else:
            print(f"Frame {i}: Low Performance")
    
    # Step 6: Calculate and display overall performance summary
    calculate_average_performance(predictions)


