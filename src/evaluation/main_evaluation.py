from predict_performance import predict_performance
import os

# Set base path for accessing data files relative to this script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Main evaluation function for testing trained models on new videos
def main_evaluation():
    # Configuration: Paths to input video, trained model, and fitted PCA
    video_file = os.path.join(base_path, 'data/raw/jump/jump_010.mp4')
    model_file = os.path.join(base_path, 'data/models/jump_model.pkl')
    pca_file = os.path.join(base_path, 'data/processed/jump_pca.pkl')
    
    # Execute performance prediction on the new video
    predict_performance(video_file, model_file, pca_file)

# Entry point for video evaluation pipeline
if __name__ == "__main__":
    main_evaluation()

