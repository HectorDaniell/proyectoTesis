import pandas as pd

# Función para etiquetar el desempeño basado en la altura del salto
def label_performance_jump(csv_file):
    df = pd.read_csv(csv_file)

    # Consideramos las columnas que representan los tobillos (landmarks)
    right_ankle_y = df['right_ankle_y']
    left_ankle_y = df['left_ankle_y']

    # Promedio de la altura de ambos tobillos (coordenada y)
    df['avg_ankle_height'] = (right_ankle_y + left_ankle_y) / 2

    # Inicializamos la columna de desempeño (performance)
    df['performance'] = 0

    # Definir umbrales basados en la altura del tobillo (cuanto menor es el valor de 'y', mayor es la altura)
    high_threshold = df['avg_ankle_height'].quantile(0.33)  # Umbral para desempeño alto (33% más alto)
    low_threshold = df['avg_ankle_height'].quantile(0.66)   # Umbral para desempeño bajo (66% más bajo)

    # Asignar etiquetas de desempeño basadas en la altura del tobillo
    df.loc[df['avg_ankle_height'] <= high_threshold, 'performance'] = 1  # Alto desempeño
    df.loc[(df['avg_ankle_height'] > high_threshold) & (df['avg_ankle_height'] <= low_threshold), 'performance'] = 2  # Desempeño moderado
    df.loc[df['avg_ankle_height'] > low_threshold, 'performance'] = 3  # Bajo desempeño

    # Guardar el archivo CSV etiquetado
    labeled_csv = csv_file.replace('_landmarks.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    print(f"Etiquetas de salto guardadas en {labeled_csv}")

    return labeled_csv
