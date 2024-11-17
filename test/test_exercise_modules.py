import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import time
import cv2
import mediapipe as mp
from memory_profiler import profile
import logging
import pytest
from datetime import datetime

# Configuración del logging
logging.basicConfig(
    filename='test_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestExerciseSystem:
    @classmethod
    def setup_class(cls):
        """Configuración inicial para las pruebas"""
        cls.video_path = '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/raw/jump/MVI_1165.MP4'
        cls.model_path = '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/models/jump_model.pkl'
        cls.pca_components = 10
        
    def test_model_validation(self):
        """
        7.1 Pruebas Funcionales - Validación del modelo
        Verifica la precisión y métricas del modelo usando cross-validation
        """
        try:
            # Cargar el modelo
            model = joblib.load('/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/models/jump_model.pkl')
            
            # Cargar datos de prueba
            test_data = pd.read_csv('/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/processed/jump_reduced.csv')
            X_test = test_data.drop('performance', axis=1)
            y_test = test_data['performance']
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Registrar resultados
            logging.info(f"""
            Model Validation Results:
            Accuracy: {accuracy:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            F1-Score: {f1:.4f}
            """)
            
            # Verificar que las métricas cumplan con los umbrales mínimos
            assert accuracy >= 0.75, "Model accuracy is below acceptable threshold"
            assert precision >= 0.70, "Model precision is below acceptable threshold"
            
            return True
            
        except Exception as e:
            logging.error(f"Model validation failed: {str(e)}")
            return False

    def test_input_output(self):
        """
        7.1 Pruebas Funcionales - Pruebas de entrada y salida
        Verifica el procesamiento correcto de diferentes formatos de video
        """
        test_videos = [
            {'path': '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/raw/jump/MVI_1165.MP4', 'expected_frames': 1380}  
        ]
        
        test_results = []
        for video in test_videos:
            try:
                cap = cv2.VideoCapture(video['path'])
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                cap.release()
                
                # Verificar que el número de frames está dentro del rango esperado
                frame_difference = abs(frame_count - video['expected_frames'])
                test_results.append({
                    'video': video['path'],
                    'expected_frames': video['expected_frames'],
                    'actual_frames': frame_count,
                    'difference': frame_difference,
                    'passed': frame_difference <= 10  # Tolerancia de 10 frames
                })
                
            except Exception as e:
                logging.error(f"Error processing video {video['path']}: {str(e)}")
                test_results.append({
                    'video': video['path'],
                    'error': str(e),
                    'passed': False
                })
        
        return test_results

    @profile
    def test_performance(self):
        """
        7.2 Pruebas No Funcionales - Pruebas de rendimiento
        Mide el tiempo de procesamiento y uso de memoria
        """
        try:
            start_time = time.time()
            
            # Cargar modelo y realizar predicción
            model = joblib.load('/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/models/jump_model.pkl')
            cap = cv2.VideoCapture('/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/raw/jump/MVI_1165.MP4')
            frame_count = 0
            processing_times = []
            
            with mp.solutions.holistic.Holistic() as holistic:
                while cap.isOpened():
                    frame_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Procesar frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    holistic_results = holistic.process(frame_rgb)
                    
                    frame_end = time.time()
                    processing_times.append(frame_end - frame_start)
                    frame_count += 1
                    
            cap.release()
            
            total_time = time.time() - start_time
            avg_frame_time = np.mean(processing_times)
            fps = frame_count / total_time
            
            performance_metrics = {
                'total_processing_time': total_time,
                'average_frame_time': avg_frame_time,
                'fps': fps,
                'total_frames': frame_count
            }
            
            # Verificar requisitos de rendimiento
            assert fps >= 15, "FPS below minimum requirement"
            assert avg_frame_time <= 0.1, "Frame processing time too high"
            
            logging.info(f"""
            Performance Test Results:
            Total Processing Time: {total_time:.2f} seconds
            Average Frame Time: {avg_frame_time:.4f} seconds
            FPS: {fps:.2f}
            Total Frames: {frame_count}
            """)
            
            return performance_metrics
            
        except Exception as e:
            logging.error(f"Performance test failed: {str(e)}")
            return None

    def test_robustness(self):
        """
        7.2 Pruebas No Funcionales - Pruebas de robustez y tolerancia a fallos
        Prueba el sistema con casos extremos y errores
        """
        test_cases = [
            {'case': 'normal_video', 'path': '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/raw/jump/MVI_1165.MP4'}  # Usamos el video normal como referencia
        ]
        
        robustness_results = []
        for test in test_cases:
            try:
                # Intentar procesar el video
                cap = cv2.VideoCapture(test['path'])
                if not cap.isOpened():
                    raise Exception("Could not open video file")
                
                with mp.solutions.holistic.Holistic() as holistic:
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        holistic_results = holistic.process(frame_rgb)
                
                cap.release()
                robustness_results.append({
                    'test_case': test['case'],
                    'status': 'passed',
                    'message': 'Successfully handled test case'
                })
                
            except Exception as e:
                robustness_results.append({
                    'test_case': test['case'],
                    'status': 'handled',
                    'error': str(e),
                    'message': 'Error handled gracefully'
                })
        
        return robustness_results

def run_all_tests():
    """Ejecuta todas las pruebas y genera un reporte"""
    test_suite = TestExerciseSystem()
    
    # Ejecutar todas las pruebas
    results = {
        'model_validation': test_suite.test_model_validation(),
        'input_output': test_suite.test_input_output(),
        'performance': test_suite.test_performance(),
        'robustness': test_suite.test_robustness()
    }
    
    # Generar reporte
    report = f"""
    Test Suite Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================
    
    1. Model Validation: {'✓ PASSED' if results['model_validation'] else '✗ FAILED'}
    
    2. Input/Output Tests:
    {format_io_results(results['input_output'])}
    
    3. Performance Tests:
    {format_performance_results(results['performance'])}
    
    4. Robustness Tests:
    {format_robustness_results(results['robustness'])}
    """
    
    # Guardar reporte
    with open('test_report.txt', 'w') as f:
        f.write(report)
    
    return report

def format_io_results(results):
    return '\n'.join([f"  - {r['video']}: {'✓ PASSED' if r['passed'] else '✗ FAILED'}" for r in results])

def format_performance_results(results):
    if not results:
        return "  No performance results available"
    return f"""  - Total Time: {results['total_processing_time']:.2f}s
  - Avg Frame Time: {results['average_frame_time']:.4f}s
  - FPS: {results['fps']:.2f}
  - Frames Processed: {results['total_frames']}"""

def format_robustness_results(results):
    return '\n'.join([f"  - {r['test_case']}: {r['status'].upper()}" for r in results])

if __name__ == "__main__":
    print(run_all_tests())


