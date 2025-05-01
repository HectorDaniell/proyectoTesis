from predict_performance import predict_performance
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Evaluar nuevos videos usando un modelo entrenado
def main_evaluation():
    video_file = os.path.join(base_path, f'data/raw/jump/jump_006.mp4')  # Video de entrada para evaluación
    model_file = os.path.join(base_path, f'data/models/XGBoost_model.pkl')  # Ruta del modelo entrenado
    pca_components = 10  # Número de componentes de PCA usados en el entrenamiento
    
    # Hacer predicción
    predict_performance(video_file, model_file, pca_components)

# Ejecutar la evaluación del video
if __name__ == "__main__":
    main_evaluation()