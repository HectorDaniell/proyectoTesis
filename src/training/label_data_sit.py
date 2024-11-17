import pandas as pd

# Función para etiquetar el desempeño para el ejercicio de saltar
def label_performance_sit(csv_file):
    df = pd.read_csv(csv_file)
    df['performance'] = 0  # Inicializar la columna
    
    # Etiquetar el desempeño de saltar basado en alguna métrica específica
    df.loc[df.index < 30, 'performance'] = 1  # Alto (primeras 30 filas)
    df.loc[(df.index >= 30) & (df.index < 60), 'performance'] = 2  # Moderado
    df.loc[df.index >= 60, 'performance'] = 3  # Bajo (últimas filas)
    
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Etiquetas de sentarse guardadas en {labeled_csv}")
    
    return labeled_csv
