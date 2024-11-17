import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Función para procesar un video y extraer puntos clave
def process_video(video_path, exercise_name, combined_df):
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        landmark_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Extraer los puntos clave (landmarks)
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Guardamos las coordenadas x, y, z de cada landmark
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmark_data.append(frame_landmarks)
        
        cap.release()

    # Definir los nombres de las columnas de acuerdo a los landmarks de MediaPipe
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

    column_names = []
    for part in body_parts:
        column_names.append(f"{part}_x")  # Coordenada x
        column_names.append(f"{part}_y")  # Coordenada y 
        column_names.append(f"{part}_z")  # Coordenada z

    # Convertir los puntos clave a un DataFrame con nombres descriptivos
    df = pd.DataFrame(landmark_data, columns=column_names)
    
    # Concatenar con el DataFrame combinado
    return pd.concat([combined_df, df], ignore_index=True)
