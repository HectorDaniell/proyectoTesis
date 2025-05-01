import pandas as pd
import numpy as np

# Función para etiquetar el desempeño basado en el ejercicio de sentarse
def label_performance_sit(csv_file):
    df = pd.read_csv(csv_file)
    
    # 1. Evaluar control postural (alineación columna-cabeza)
    # Vector de columna (desde cadera a hombros)
    spine_vector_x = df['left_shoulder_x'] + df['right_shoulder_x'] - df['left_hip_x'] - df['right_hip_x']
    spine_vector_y = df['left_shoulder_y'] + df['right_shoulder_y'] - df['left_hip_y'] - df['right_hip_y']
    
    # Ángulo con respecto a la vertical (debería estar cerca de 0° en posición sentada óptima)
    df['spine_angle'] = np.abs(np.arctan2(spine_vector_x, spine_vector_y))
    df['posture_score'] = 1 - df['spine_angle'] / np.pi  # Normalizado 0-1 (1 = vertical)
    
    # 2. Evaluar simetría (diferencia entre lados izquierdo y derecho)
    hip_symmetry = 1 - np.abs(df['left_hip_y'] - df['right_hip_y'])
    shoulder_symmetry = 1 - np.abs(df['left_shoulder_y'] - df['right_shoulder_y'])
    df['symmetry_score'] = (hip_symmetry + shoulder_symmetry) / 2
    
    # 3. Detectar transiciones (sentarse/levantarse) para evaluar suavidad
    hip_height = (df['left_hip_y'] + df['right_hip_y']) / 2
    hip_velocity = hip_height.diff().rolling(window=5).mean()
    
    # Identificar transiciones (cambios sostenidos en altura de cadera)
    df['is_transition'] = (hip_velocity.abs() > hip_velocity.abs().quantile(0.7)).astype(int)
    
    # Calcular suavidad en transiciones (menor aceleración = más suave)
    df['transition_smoothness'] = 1 - hip_velocity.diff().abs() / hip_velocity.abs().max()
    
    # Combinar métricas en un puntaje total
    df['total_score'] = (
        df['posture_score'].fillna(0) * 0.4 +            # 40% postura
        df['symmetry_score'].fillna(0) * 0.3 +           # 30% simetría
        df['transition_smoothness'].fillna(0) * 0.3      # 30% suavidad de transición
    )
    
    # Clasificación final
    high_threshold = df['total_score'].quantile(0.67)
    low_threshold = df['total_score'].quantile(0.33)
    
    df['performance'] = 2  # Valor predeterminado: moderado
    df.loc[df['total_score'] >= high_threshold, 'performance'] = 1  # Alto
    df.loc[df['total_score'] <= low_threshold, 'performance'] = 3   # Bajo
    
    # Guardar resultados
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Etiquetas de sentarse guardadas en {labeled_csv}")
    
    return labeled_csv
