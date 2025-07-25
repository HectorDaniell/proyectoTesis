from predict_performance import predict_performance
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Evaluate new videos using a trained model
def main_evaluation():
    video_file = os.path.join(base_path, f'data/raw/jump/jump_006.mp4')  # Input video for evaluation
    model_file = os.path.join(base_path, f'data/models/XGBoost_model.pkl')  # Path to the trained model
    pca_components = 10  # Number of PCA components used in training
    
    # Make prediction
    predict_performance(video_file, model_file, pca_components)

# Run video evaluation
if __name__ == "__main__":
    main_evaluation()
