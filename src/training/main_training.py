import os
import pandas as pd
from process_videos import process_video
from pca_reduction import apply_pca
from train_model import train_and_evaluate_model

# Import labeling modules for each exercise type
#from label_data_abdominales import label_performance_abdominales
from label_data_jump import label_performance_jump
from label_data_crawl import label_performance_crawl
from label_data_sit import label_performance_sit
from label_data_throw import label_performance_throw

# Set base path for accessing data files relative to this script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


# Function to select appropriate labeling function based on exercise type
def select_labeling_function(exercise_name):
    """
    Select the appropriate performance labeling function based on exercise type.
    
    Args:
        exercise_name (str): Name of the exercise (jump, crawl, sit, ball throwing)
        
    Returns:
        function: Appropriate labeling function for the exercise
        
    Raises:
        ValueError: If no labeling function is available for the exercise
    """
    if exercise_name == 'jump':
        return label_performance_jump
    elif exercise_name == 'crawl':
        return label_performance_crawl
    elif exercise_name == 'sit':
        return label_performance_sit
    elif exercise_name == 'throw':
        return label_performance_throw
    else:
        raise ValueError(f"No labeling function available for exercise {exercise_name}")

# Main function that manages the complete training pipeline for a specific exercise
def main_training(exercise_name):
    """
    Complete training pipeline for exercise performance classification.
    
    This function orchestrates the entire training process:
    1. Video processing and landmark extraction
    2. Performance labeling based on exercise-specific criteria
    3. Dimensionality reduction using PCA
    4. Model training and evaluation
    
    Args:
        exercise_name (str): Name of the exercise to train for
    """
    # Configure paths for input videos and output files
    raw_videos_path = os.path.join(base_path, f'data/raw/{exercise_name}/')  # Exercise-specific folder
    print(f"Looking for videos in: {raw_videos_path}")
    processed_folder = os.path.join(base_path, 'data/processed')  # Folder to save processed files

    # Get all MP4 videos in the exercise folder
    videos = [f for f in os.listdir(raw_videos_path) if f.endswith('.mp4')]
    print(f"Found {len(videos)} videos to process")

    # Initialize empty DataFrame to combine key points from all videos
    combined_df = pd.DataFrame()

    # Process each video and combine landmark data
    for video in videos:
        video_path = os.path.join(raw_videos_path, video)
        print(f"Processing video: {video_path}")
        combined_df = process_video(video_path, exercise_name, combined_df)

    # Save combined landmark data for all videos
    combined_csv = os.path.join(processed_folder, f'{exercise_name}_landmarks.csv')
    combined_df.to_csv(combined_csv, index=False)
    print(f"Combined landmarks saved in {combined_csv}")
    
    # Select appropriate labeling function based on exercise type
    label_function = select_labeling_function(exercise_name)
    
    # Step 1: Label performance based on exercise-specific criteria
    print(f"Labeling performance for {exercise_name} exercise...")
    labeled_csv = label_function(combined_csv)
    
    # Step 2: Apply PCA for dimensionality reduction
    print("Applying PCA dimensionality reduction...")
    reduced_csv = apply_pca(labeled_csv)
    
    # Step 3: Train and evaluate the machine learning model
    print("Training and evaluating model...")
    train_and_evaluate_model(reduced_csv, exercise_name)

# Entry point: Execute training pipeline for jump exercise
if __name__ == "__main__":
    main_training('throw') 
