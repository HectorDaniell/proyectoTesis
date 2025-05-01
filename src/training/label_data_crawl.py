import pandas as pd
import numpy as np

# Función para etiquetar el desempeño basado en el ejercicio de gateo
def label_performance_crawl(csv_file):
    df = pd.read_csv(csv_file)
    
    # 1. Calcular la coordinación diagonal (correlación entre movimientos cruzados)
    # Extraer movimientos de muñecas y rodillas
    right_wrist_x = df['right_wrist_x'].diff()  # Cambio en posición
    left_knee_x = df['left_knee_x'].diff()      # Cambio en posición
    left_wrist_x = df['left_wrist_x'].diff()    # Cambio en posición
    right_knee_x = df['right_knee_x'].diff()    # Cambio en posición
    
    # Calcular correlación entre movimientos cruzados (debe ser negativa en gateo correcto)
    df['right_left_coordination'] = right_wrist_x * left_knee_x  
    df['left_right_coordination'] = left_wrist_x * right_knee_x
    df['coordination_score'] = (df['right_left_coordination'] + df['left_right_coordination']).rolling(window=10).mean()
    
    # 2. Evaluar estabilidad de cadera (variación en altura de caderas)
    hip_height = (df['right_hip_y'] + df['left_hip_y']) / 2
    df['hip_stability'] = 1 - hip_height.rolling(window=10).std()  # Menor variación = mayor estabilidad
    
    # 3. Calcular fluidez (cambios suaves vs bruscos en velocidad)
    hip_velocity = hip_height.diff().abs()
    df['movement_fluidity'] = 1 - hip_velocity.rolling(window=10).std() / hip_velocity.rolling(window=10).mean()
    
    # Combinar métricas en un puntaje total
    df['total_score'] = (
        df['coordination_score'].abs().fillna(0) * 0.4 +  # 40% coordinación
        df['hip_stability'].fillna(0) * 0.3 +             # 30% estabilidad
        df['movement_fluidity'].fillna(0) * 0.3           # 30% fluidez
    )
    
    # Clasificación basada en el puntaje total
    high_threshold = df['total_score'].quantile(0.67)
    low_threshold = df['total_score'].quantile(0.33)
    
    df['performance'] = 2  # Valor predeterminado: moderado
    df.loc[df['total_score'] >= high_threshold, 'performance'] = 1  # Alto
    df.loc[df['total_score'] <= low_threshold, 'performance'] = 3   # Bajo
    
    # Guardar resultados
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Etiquetas de gateo guardadas en {labeled_csv}")
    
    return labeled_csv
