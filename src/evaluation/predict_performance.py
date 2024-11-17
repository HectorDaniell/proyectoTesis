import cv2
import mediapipe as mp
import pandas as pd
import joblib
from sklearn.decomposition import PCA

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Función para procesar el video y extraer los puntos clave
def process_new_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        landmark_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Extraer los puntos clave
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmark_data.append(frame_landmarks)
        
        cap.release()

    # Convertir los puntos clave a un DataFrame
    df = pd.DataFrame(landmark_data)
    return df


# Función para calcular el promedio de desempeño
def calcular_desempeno_promedio(predictions):
    # Asignar valores numéricos a los desempeños: Alto=1, Moderado=2, Bajo=3
    desempeno_numerico = [1 if p == 1 else 2 if p == 2 else 3 for p in predictions]
    
    # Calcular el promedio
    promedio = sum(desempeno_numerico) / len(desempeno_numerico)
    
    # Determinar el desempeño general en función del promedio
    if promedio <= 1.5:
        desempeno_general = "Alto"
    elif promedio <= 2.5:
        desempeno_general = "Moderado"
    else:
        desempeno_general = "Bajo"
    
    print(f"\nPromedio de desempeño: {promedio:.2f}")
    print(f"Desempeño general del ejercicio: {desempeno_general}")



# Función para cargar el modelo y hacer predicciones sobre un nuevo video
def predict_performance(video_path, model_path, pca_components):
    # 1. Procesar el nuevo video
    new_data_df = process_new_video(video_path)

    # 2. Aplicar la misma reducción de dimensionalidad (PCA) que en el entrenamiento
    pca = PCA(n_components=pca_components)
    new_data_reduced = pca.fit_transform(new_data_df)

    # 3. Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # 4. Hacer predicciones
    predictions = model.predict(new_data_reduced)

    # 5. Mostrar resultados
    for i, pred in enumerate(predictions):
        if pred == 1:
            print(f"Frame {i}: Desempeño Alto")
        elif pred == 2:
            print(f"Frame {i}: Desempeño Moderado")
        else:
            print(f"Frame {i}: Desempeño Bajo")
    # 6. Calcular y mostrar el promedio de desempeño general
    calcular_desempeno_promedio(predictions)


