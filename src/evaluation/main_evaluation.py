from predict_performance import predict_performance
import os

# Set base path for accessing data files relative to this script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Main evaluation function for testing trained models on new videos
def main_evaluation():
    # Configuration: Paths to input video and trained model
    video_file = os.path.join(base_path, f'data/raw/throw/throw_009.mp4')  # Input video for evaluation
    model_file = os.path.join(base_path, f'data/models/throw_model.pkl')  # Path to the trained model
    pca_components = 10  # Number of PCA components used in training (must match training configuration)
    
    # Execute performance prediction on the new video
    predict_performance(video_file, model_file, pca_components)

# Entry point for video evaluation pipeline
if __name__ == "__main__":
    main_evaluation()

