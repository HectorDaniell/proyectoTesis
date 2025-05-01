import pandas as pd

def extract_landmarks_from_frame(results):
    # Inicializamos la lista de puntos clave
    landmarks = []

    # Si hay landmarks del cuerpo
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

    # Almacenar los landmarks en un archivo CSV
    df = pd.DataFrame([landmarks])
    df.to_csv('landmarks_output.csv', mode='a', header=False, index=False)