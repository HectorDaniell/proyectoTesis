import cv2
import mediapipe as mp
import pandas as pd
import joblib
from sklearn.decomposition import PCA

# Initialize MediaPipe components for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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
                landmark_data.append(frame_landmarks)
        
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
def predict_performance(video_path, model_path, pca_components):
    """
    Complete pipeline for predicting exercise performance on a new video.
    
    Args:
        video_path (str): Path to input video file
        model_path (str): Path to trained model file (.pkl)
        pca_components (int): Number of PCA components used in training
    """
    # Step 1: Process the new video to extract pose landmarks
    print("Processing video and extracting pose landmarks...")
    new_data_df = process_new_video(video_path)

    # Step 2: Apply the same dimensionality reduction (PCA) as in training
    print("Applying PCA dimensionality reduction...")
    pca = PCA(n_components=pca_components)
    new_data_reduced = pca.fit_transform(new_data_df)

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


