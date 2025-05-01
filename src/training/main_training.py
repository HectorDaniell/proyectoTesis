import os
import pandas as pd
from process_videos import process_video
from pca_reduction import apply_pca
from train_model import train_and_evaluate_model

# Importar los módulos de etiquetado para cada ejercicio
#from label_data_abdominales import label_performance_abdominales
from label_data_jump import label_performance_jump
from label_data_crawl import label_performance_crawl
from label_data_sit import label_performance_sit
from label_data_ball_throwing import label_performance_ball_throwing
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


# Función para seleccionar la función de etiquetado según el ejercicio
def select_labeling_function(exercise_name):
    if exercise_name == 'jump':
        return label_performance_jump
    elif exercise_name == 'crawl':
        return label_performance_crawl
    elif exercise_name == 'sit':
        return label_performance_sit
    elif exercise_name == 'ball throwing':
        return label_performance_ball_throwing
    else:
        raise ValueError(f"No hay una función de etiquetado para el ejercicio {exercise_name}")

# Función principal que gestiona el entrenamiento para un ejercicio específico
def main_training(exercise_name):
    #raw_videos_path = f'/data/raw/{exercise_name}/'  # Carpeta específica del ejercicio
    raw_videos_path = os.path.join(base_path, f'data/raw/{exercise_name}/')# Carpeta específica del ejercicio
    print(f"Buscando videos en: {raw_videos_path}")
    processed_folder = os.path.join(base_path, 'data/processed')  # Carpeta para guardar los archivos procesados

    # Obtener todos los videos en la carpeta del ejercicio
    videos = [f for f in os.listdir(raw_videos_path) if f.endswith('.mp4')]

    # Inicializar un DataFrame vacío para combinar los puntos clave
    combined_df = pd.DataFrame()

    # Procesar cada video y combinar los datos en un único DataFrame
    for video in videos:
        video_path = os.path.join(raw_videos_path, video)
        print(f"Procesando el video: {video_path}")
        combined_df = process_video(video_path, exercise_name, combined_df)

    # Guardar los puntos clave combinados
    combined_csv = os.path.join(processed_folder, f'{exercise_name}_landmarks.csv')
    combined_df.to_csv(combined_csv, index=False)
    print(f"Landmarks combinados guardados en {combined_csv}")
    
    # Seleccionar la función de etiquetado adecuada según el ejercicio
    label_function = select_labeling_function(exercise_name)
    
    # 1. Etiquetar el desempeño
    labeled_csv = label_function(combined_csv)
    
    # 2. Aplicar PCA (reducción de dimensionalidad)
    reduced_csv = apply_pca(labeled_csv)
    
    # 3. Entrenar y guardar el modelo
    train_and_evaluate_model(reduced_csv)

# Llamada a la función principal para cada ejercicio
main_training('jump') 
