from predict_performance import predict_performance

# Evaluar nuevos videos usando un modelo entrenado
def main_evaluation():
    video_file = '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/raw/jump/MVI_1192.MP4'  # Video de entrada para evaluación
    model_file = '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/models/jump_model.pkl'  # Ruta del modelo entrenado
    pca_components = 10  # Número de componentes de PCA usados en el entrenamiento
    
    # Hacer predicción
    predict_performance(video_file, model_file, pca_components)

# Ejecutar la evaluación del video
if __name__ == "__main__":
    main_evaluation()