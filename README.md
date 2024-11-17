# Sistema Basado en Visión Computacional para Evaluación de Motricidad Gruesa

## Descripción
Este proyecto implementa un sistema que utiliza visión computacional y aprendizaje automático para evaluar el desempeño motor en ejercicios realizados por niños con síndrome de Down. Utiliza herramientas como MediaPipe, OpenCV, PCA, y Random Forest.

## Requisitos Previos
- Python 3.9.6
- Git (opcional para clonar el repositorio)
- Cámara compatible para pruebas (opcional)

## Configuración del Entorno

### 1. Clonar el Repositorio (opcional)
Si deseas clonar el proyecto desde GitHub:
git clone https://github.com/HectorDaniell/proyectoTesis.git
cd repo-nombre

2. Crear un Entorno Virtual
Para evitar conflictos de dependencias, se recomienda usar un entorno virtual:
python -m venv .venv

3. Activar el Entorno Virtual
En Windows:
.venv\Scripts\activate

En macOS/Linux:
source .venv/bin/activate

4. Actualizar pip
Asegúrate de tener la última versión de pip:
pip install --upgrade pip

6. Instalar Dependencias
Instala las dependencias del proyecto:

MediaPipe:  
pip install mediapipe

OpenCV:
pip install opencv-contrib-python

Pandas:
pip install pandas

Scikit-learn:
pip install scikit-learn

Matplotlib:
pip install matplotlib

Seaborn:
pip install seaborn

Joblib:
pip install joblib



## Uso del Proyecto
1. Entrenamiento del Modelo
Para entrenar el modelo con un conjunto de videos específicos:
python main_train.py

2. Evaluación de Nuevos Videos
Para evaluar un nuevo video con el modelo entrenado:
python main_evaluation.py


Contacto
Para dudas o mejoras, puedes contactarme en:

Email: [hector.oviedo1312@gmail.com]
GitHub: [https://github.com/HectorDaniell]
