import pandas as pd
import numpy as np

# Función para etiquetar el desempeño basado en el lanzamiento de pelota
def label_performance_ball_throwing(csv_file):
    df = pd.read_csv(csv_file)
    
    # 1. Evaluar secuencia cinemática (orden correcto de activación articular)
    # Calcular velocidades de movimiento
    trunk_rotation = np.arctan2(df['right_shoulder_x'] - df['left_shoulder_x'], 
                               df['right_shoulder_y'] - df['left_shoulder_y']).diff()
    shoulder_movement = (df['right_shoulder_x'].diff()**2 + df['right_shoulder_y'].diff()**2) ** 0.5
    elbow_movement = (df['right_elbow_x'].diff()**2 + df['right_elbow_y'].diff()**2) ** 0.5
    wrist_movement = (df['right_wrist_x'].diff()**2 + df['right_wrist_y'].diff()**2) ** 0.5
    
    # Suavizar señales para reducir ruido
    window = 5
    trunk_rotation_smooth = trunk_rotation.abs().rolling(window=window).mean().fillna(0)
    shoulder_movement_smooth = shoulder_movement.rolling(window=window).mean().fillna(0)
    elbow_movement_smooth = elbow_movement.rolling(window=window).mean().fillna(0)
    wrist_movement_smooth = wrist_movement.rolling(window=window).mean().fillna(0)
    
    # Encontrar picos de actividad para cada articulación
    threshold_trunk = trunk_rotation_smooth.quantile(0.8)
    threshold_shoulder = shoulder_movement_smooth.quantile(0.8)
    threshold_elbow = elbow_movement_smooth.quantile(0.8)
    threshold_wrist = wrist_movement_smooth.quantile(0.8)
    
    df['trunk_active'] = (trunk_rotation_smooth > threshold_trunk).astype(int)
    df['shoulder_active'] = (shoulder_movement_smooth > threshold_shoulder).astype(int)
    df['elbow_active'] = (elbow_movement_smooth > threshold_elbow).astype(int)
    df['wrist_active'] = (wrist_movement_smooth > threshold_wrist).astype(int)
    
    # Evaluar secuencia de proximal a distal (tronco → hombro → codo → muñeca)
    df['sequencing_score'] = 0.0
    
    # Para cada ventana de frames, verificar si la secuencia es correcta
    for i in range(len(df) - window):
        seq_window = df.iloc[i:i+window]
        # Si hay activación en todas las articulaciones
        if (seq_window['trunk_active'].sum() > 0 and seq_window['shoulder_active'].sum() > 0 and
            seq_window['elbow_active'].sum() > 0 and seq_window['wrist_active'].sum() > 0):
            
            # Encontrar el primer frame de activación para cada articulación
            first_trunk = seq_window['trunk_active'].idxmax()
            first_shoulder = seq_window['shoulder_active'].idxmax()
            first_elbow = seq_window['elbow_active'].idxmax()
            first_wrist = seq_window['wrist_active'].idxmax()
            
            # Verificar si la secuencia es correcta (ideal: trunk < shoulder < elbow < wrist)
            correct_sequence = (first_trunk <= first_shoulder and 
                                first_shoulder <= first_elbow and 
                                first_elbow <= first_wrist)
            
            # Asignar puntaje basado en la secuencia
            df.loc[i:i+window, 'sequencing_score'] = 1.0 if correct_sequence else 0.5
    
    # 2. Amplitud de movimiento (distancia máxima de la muñeca)
    # Calcular la posición de reposo (promedio de los primeros frames)
    rest_wrist_x = df['right_wrist_x'][:10].mean()
    rest_wrist_y = df['right_wrist_y'][:10].mean()
    
    # Calcular desplazamiento máximo desde la posición de reposo
    df['wrist_displacement'] = ((df['right_wrist_x'] - rest_wrist_x)**2 + 
                               (df['right_wrist_y'] - rest_wrist_y)**2) ** 0.5
    max_displacement = df['wrist_displacement'].max()
    df['amplitude_score'] = df['wrist_displacement'] / max_displacement if max_displacement > 0 else 0
    
    # 3. Estabilidad postural durante el lanzamiento
    hip_movement = ((df['right_hip_x'].diff()**2 + df['right_hip_y'].diff()**2) ** 0.5).rolling(window=5).mean()
    hip_movement_max = hip_movement.max()
    df['stability_score'] = 1 - hip_movement / hip_movement_max if hip_movement_max > 0 else 1
    
    # Combinar métricas en un puntaje total
    df['total_score'] = (
        df['sequencing_score'].fillna(0) * 0.4 +  # 40% secuencia de movimiento
        df['amplitude_score'].fillna(0) * 0.3 +   # 30% amplitud
        df['stability_score'].fillna(1) * 0.3     # 30% estabilidad
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
    print(f"Etiquetas de lanzamiento de pelota guardadas en {labeled_csv}")
    
    return labeled_csv
