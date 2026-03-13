# Documentación del Sistema de Evaluación de Habilidades Motoras Gruesas

---

## Parte 1: Explicación en Términos Simples

### ¿Qué hace este sistema?

Imagina a un entrenador o terapeuta que observa a un niño mientras realiza ejercicios físicos: saltar, gatear, sentarse/levantarse o lanzar una pelota. El entrenador mira los movimientos del niño y decide si el niño lo hizo bien, regular o si necesita mejorar.

Este sistema hace exactamente lo mismo, pero de forma automática. En lugar de un entrenador humano, usa una cámara y un programa de computadora que "ve" los movimientos del niño en un video y los evalúa.

### ¿Cómo "ve" el sistema los movimientos?

Cuando el sistema recibe un video de un niño haciendo un ejercicio, lo primero que hace es identificar las partes del cuerpo en cada imagen del video (cada "frame"). El sistema detecta **33 puntos clave del cuerpo**, como:

- La nariz
- Los ojos y las orejas
- Los hombros, codos y muñecas
- Las caderas, rodillas y tobillos
- Los pies

Estos puntos se llaman **landmarks** (puntos de referencia). Para cada punto, el sistema sabe su posición exacta en la imagen: qué tan a la izquierda/derecha está y qué tan arriba/abajo está.

### ¿Qué mide en cada ejercicio?

#### Salto (Jump)

El sistema mide **qué tan alto salta el niño**. Para esto, observa la posición de los tobillos:

- Si los tobillos están muy arriba en la imagen → el niño saltó alto → **buen rendimiento**
- Si los tobillos están más abajo → el salto fue más bajo → **rendimiento bajo**

Es como si el entrenador mirara los pies del niño y evaluara la altura que alcanzan durante el salto.

#### Gateo (Crawl)

El sistema evalúa **tres aspectos** del gateo:

1. **Coordinación** (40% de la nota): ¿El niño mueve el brazo derecho junto con la rodilla izquierda y viceversa? Un gateo correcto tiene este patrón cruzado, como caminar.
2. **Estabilidad de la cadera** (30%): ¿Las caderas del niño se mantienen a una altura constante mientras gatea, o suben y bajan mucho?
3. **Fluidez del movimiento** (30%): ¿El movimiento es suave y continuo, o es brusco y entrecortado?

#### Sentarse/Levantarse (Sit)

El sistema evalúa **tres aspectos**:

1. **Postura** (40% de la nota): ¿La espalda del niño está recta? El sistema mide el ángulo de la columna respecto a la vertical.
2. **Simetría** (30%): ¿Ambos lados del cuerpo están equilibrados? Compara la posición del hombro izquierdo con el derecho, y la cadera izquierda con la derecha.
3. **Suavidad al levantarse** (30%): ¿El niño se levanta de forma suave o de forma brusca? Un movimiento suave indica mejor control motor.

#### Lanzamiento de pelota (Throw)

El sistema evalúa **tres aspectos**:

1. **Secuencia del movimiento** (40% de la nota): En un buen lanzamiento, el cuerpo se mueve en orden: primero el tronco gira, luego el hombro, después el codo y finalmente la muñeca. Es como un "efecto látigo". El sistema verifica si el niño sigue esta secuencia.
2. **Amplitud** (30%): ¿Qué tanto se extiende el brazo durante el lanzamiento? Más extensión = mejor rendimiento.
3. **Equilibrio** (30%): ¿El niño mantiene el equilibrio mientras lanza, o se tambalea? Se mide observando el movimiento de las caderas.

### ¿Cómo decide si el rendimiento es Alto, Moderado o Bajo?

El sistema combina todas las mediciones de un ejercicio en una puntuación total. Luego divide a todos los niños en tres grupos iguales:

- **Alto rendimiento (1)**: El mejor tercio (top 33%)
- **Rendimiento moderado (2)**: El tercio del medio
- **Bajo rendimiento (3)**: El tercio inferior

Es como si el entrenador ordenara a todos los niños de mejor a peor y los dividiera en tres grupos.

### ¿Cómo se "entrena" el sistema?

Antes de poder evaluar a un niño nuevo, el sistema necesita "aprender" qué es un buen movimiento y qué no lo es:

1. **Se le muestran muchos videos** de niños haciendo el ejercicio
2. **Identifica los puntos del cuerpo** en cada video
3. **Calcula las mediciones** específicas del ejercicio
4. **Clasifica cada frame** como alto, moderado o bajo rendimiento
5. **Aprende los patrones** usando un algoritmo de inteligencia artificial (Random Forest)

Una vez entrenado, cuando recibe un video nuevo, puede predecir automáticamente cómo fue el rendimiento del niño.

### ¿Cómo evalúa un video nuevo?

1. Recibe el video del niño realizando el ejercicio
2. Identifica los puntos del cuerpo en cada frame
3. Prepara los datos de la misma forma que durante el entrenamiento
4. El modelo de IA analiza los datos y da una predicción por cada frame
5. Promedia todas las predicciones para dar una evaluación global: Alto, Moderado o Bajo

---

## Parte 2: Explicación Técnica

### 1. Pipeline de Extracción de Features

#### 1.1 Detección de Pose con MediaPipe Holistic

El sistema usa **MediaPipe Holistic** (versión 0.10.5) con la siguiente configuración:

```python
mp.solutions.holistic.Holistic(
    static_image_mode=False,   # Modo video (optimizado para secuencias)
    model_complexity=1          # Complejidad media (balance precisión/velocidad)
)
```

**Proceso frame a frame:**

1. Captura del frame con OpenCV: `cv2.VideoCapture(video_path)`
2. Conversión de color: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
3. Inferencia: `holistic.process(frame_rgb)`
4. Extracción de `results.pose_landmarks.landmark` → 33 landmarks

#### 1.2 Landmarks Extraídos

MediaPipe Pose detecta 33 landmarks del cuerpo. Cada landmark tiene tres coordenadas:

| Coordenada | Rango | Descripción |
|------------|-------|-------------|
| `x` | 0.0 – 1.0 | Posición horizontal normalizada (0=izquierda, 1=derecha) |
| `y` | 0.0 – 1.0 | Posición vertical normalizada (0=arriba, 1=abajo) |
| `z` | ~-1.0 – 1.0 | Profundidad relativa al centro de la cadera |

**Total de features por frame:** 33 landmarks × 3 coordenadas = **99 features**.

**Lista completa de landmarks:**

| Índice | Nombre | Índice | Nombre |
|--------|--------|--------|--------|
| 0 | nose | 17 | left_knee |
| 1 | left_eye_inner | 18 | right_knee |
| 2 | left_eye | 19 | left_ankle |
| 3 | left_eye_outer | 20 | right_ankle |
| 4 | right_eye_inner | 21 | left_heel |
| 5 | right_eye | 22 | right_heel |
| 6 | right_eye_outer | 23 | left_foot_index |
| 7 | left_ear | 24 | right_foot_index |
| 8 | right_ear | 25 | left_pinky |
| 9 | left_shoulder | 26 | right_pinky |
| 10 | right_shoulder | 27 | left_index |
| 11 | left_elbow | 28 | right_index |
| 12 | right_elbow | 29 | left_thumb |
| 13 | left_wrist | 30 | right_thumb |
| 14 | right_wrist | 31 | left_foot |
| 15 | left_hip | 32 | right_foot |
| 16 | right_hip | | |

#### 1.3 Formato de Salida

El DataFrame resultante tiene la estructura:

```
nose_x | nose_y | nose_z | left_eye_inner_x | ... | right_foot_z
0.512  | 0.234  | -0.031 | 0.498            | ... | 0.089
0.515  | 0.231  | -0.029 | 0.501            | ... | 0.091
...
```

Cada fila representa un frame del video. El archivo se guarda como `data/processed/{exercise}_landmarks.csv`.

---

### 2. Métricas Biomecánicas por Ejercicio

#### 2.1 Salto (Jump)

**Archivo:** `src/training/label_data_jump.py`
**Función:** `label_performance_jump(csv_file)`

**Métrica única: Altura promedio de tobillos**

```
avg_ankle_height = (right_ankle_y + left_ankle_y) / 2
```

Dado que en MediaPipe el eje Y va de 0 (arriba) a 1 (abajo), valores menores de `avg_ankle_height` indican saltos más altos.

**Clasificación por percentiles:**

| Condición | Etiqueta | Significado |
|-----------|----------|-------------|
| `avg_ankle_height ≤ percentil_33` | 1 | Alto rendimiento (salto alto) |
| `percentil_33 < avg_ankle_height ≤ percentil_66` | 2 | Rendimiento moderado |
| `avg_ankle_height > percentil_66` | 3 | Bajo rendimiento (salto bajo) |

**Columnas agregadas al CSV:**
- `avg_ankle_height`: altura promedio de tobillos por frame
- `performance`: etiqueta 1, 2 o 3

---

#### 2.2 Gateo (Crawl)

**Archivo:** `src/training/label_data_crawl.py`
**Función:** `label_performance_crawl(csv_file)`

**Métrica 1: Coordinación diagonal (peso 40%)**

Evalúa el patrón de movimiento cruzado (brazo derecho con pierna izquierda y viceversa):

```
right_left_coord = diff(right_wrist_x) × diff(left_knee_x)
left_right_coord = diff(left_wrist_x) × diff(right_knee_x)
coordination_score = rolling_mean(right_left_coord + left_right_coord, window=10)
```

Un valor positivo alto indica buena coordinación diagonal: ambos pares de extremidades opuestas se mueven en la misma dirección simultáneamente.

**Métrica 2: Estabilidad de cadera (peso 30%)**

```
hip_height = (right_hip_y + left_hip_y) / 2
hip_stability = 1 - rolling_std(hip_height, window=10)
```

Menor variabilidad en la altura de cadera = mayor estabilidad = mejor rendimiento.

**Métrica 3: Fluidez del movimiento (peso 30%)**

```
hip_velocity = |diff(hip_height)|
velocity_ratio = rolling_std(hip_velocity, window=10) / rolling_mean(hip_velocity, window=10)
movement_fluidity = 1 - velocity_ratio
```

Un movimiento fluido tiene velocidad constante (baja desviación estándar relativa). Movimientos bruscos tienen picos de velocidad.

**Puntuación total:**

```
total_score = |coordination_score| × 0.4 + hip_stability × 0.3 + movement_fluidity × 0.3
```

**Clasificación:** percentiles 33 y 67 de `total_score`.

---

#### 2.3 Sentarse/Levantarse (Sit)

**Archivo:** `src/training/label_data_sit.py`
**Función:** `label_performance_sit(csv_file)`

**Métrica 1: Control postural (peso 40%)**

Calcula el ángulo de la columna vertebral respecto a la vertical:

```
spine_vector_x = (left_shoulder_x + right_shoulder_x) - (left_hip_x + right_hip_x)
spine_vector_y = (left_shoulder_y + right_shoulder_y) - (left_hip_y + right_hip_y)
spine_angle = |arctan2(spine_vector_x, spine_vector_y)|
posture_score = 1 - spine_angle / π
```

Un valor de `posture_score` cercano a 1 indica columna vertical (postura perfecta). Un valor cercano a 0 indica columna horizontal.

**Métrica 2: Simetría corporal (peso 30%)**

```
hip_symmetry = 1 - |left_hip_y - right_hip_y|
shoulder_symmetry = 1 - |left_shoulder_y - right_shoulder_y|
symmetry_score = (hip_symmetry + shoulder_symmetry) / 2
```

Valores cercanos a 1 indican alta simetría entre el lado izquierdo y derecho del cuerpo.

**Métrica 3: Suavidad de transiciones (peso 30%)**

```
hip_height = (left_hip_y + right_hip_y) / 2
hip_velocity = rolling_mean(diff(hip_height), window=5)
is_transition = |hip_velocity| > percentil_70(|hip_velocity|)
hip_acceleration = |diff(hip_velocity)|
transition_smoothness = 1 - hip_acceleration / max(|hip_velocity|)
```

Se detectan momentos de transición (sentarse↔levantarse) como periodos de alta velocidad en la cadera. La suavidad se mide por la aceleración durante esos periodos: menor aceleración = transición más suave.

**Puntuación total:**

```
total_score = posture_score × 0.4 + symmetry_score × 0.3 + transition_smoothness × 0.3
```

**Clasificación:** percentiles 33 y 67 de `total_score`.

---

#### 2.4 Lanzamiento de pelota (Throw)

**Archivo:** `src/training/label_data_throw.py`
**Función:** `label_performance_throw(csv_file)`

**Métrica 1: Secuencia cinemática proximal-distal (peso 40%)**

La biomecánica correcta del lanzamiento sigue una cadena cinemática: tronco → hombro → codo → muñeca. El sistema verifica este patrón:

```
# Velocidades de cada articulación
trunk_rotation = |diff(arctan2(right_shoulder_x - left_shoulder_x, right_shoulder_y - left_shoulder_y))|
shoulder_movement = sqrt(diff(right_shoulder_x)² + diff(right_shoulder_y)²)
elbow_movement = sqrt(diff(right_elbow_x)² + diff(right_elbow_y)²)
wrist_movement = sqrt(diff(right_wrist_x)² + diff(right_wrist_y)²)

# Suavizado con ventana de 5 frames
señal_suavizada = rolling_mean(señal, window=5)

# Umbrales de activación (percentil 80)
threshold = quantile(señal_suavizada, 0.8)
joint_active = señal_suavizada > threshold

# Verificación de secuencia en ventanas de 5 frames
# Se busca: first_trunk ≤ first_shoulder ≤ first_elbow ≤ first_wrist
sequencing_score = 1.0 si secuencia correcta, 0.5 si no
```

**Métrica 2: Amplitud de movimiento (peso 30%)**

```
# Posición de reposo = promedio de los primeros 10 frames
rest_wrist_x = mean(right_wrist_x[:10])
rest_wrist_y = mean(right_wrist_y[:10])

# Desplazamiento de la muñeca respecto a reposo
wrist_displacement = sqrt((right_wrist_x - rest_wrist_x)² + (right_wrist_y - rest_wrist_y)²)
amplitude_score = wrist_displacement / max(wrist_displacement)
```

Mayor desplazamiento = mayor rango de movimiento = mejor rendimiento.

**Métrica 3: Estabilidad postural (peso 30%)**

```
hip_movement = rolling_mean(sqrt(diff(right_hip_x)² + diff(right_hip_y)²), window=5)
stability_score = 1 - hip_movement / max(hip_movement)
```

Menor movimiento de la cadera durante el lanzamiento = mejor estabilidad postural.

**Puntuación total:**

```
total_score = sequencing_score × 0.4 + amplitude_score × 0.3 + stability_score × 0.3
```

**Clasificación:** percentiles 33 y 67 de `total_score`.

---

### 3. Reducción de Dimensionalidad (PCA)

**Archivo:** `src/training/pca_reduction.py`

Tras el etiquetado, el CSV contiene las 99 columnas originales de landmarks, columnas intermedias de scoring (varían según el ejercicio) y la columna `performance`.

**Proceso:**

1. Se seleccionan **solo las 99 columnas de landmarks** (aquellas que terminan en `_x`, `_y`, `_z`), descartando las columnas intermedias de scoring y `performance`.
2. Se aplica `PCA(n_components=10)` sobre las 99 features de landmarks.
3. Se genera un DataFrame con `PC1`–`PC10` + `performance`.
4. Se guarda como `{exercise}_reduced.csv`.
5. Se serializa el objeto PCA entrenado como `{exercise}_pca.pkl` con `joblib.dump()`.

**Persistencia del PCA:** El objeto PCA se guarda en `data/processed/{exercise}_pca.pkl` para que la fase de evaluación utilice exactamente la misma transformación. En evaluación se carga con `joblib.load()` y se aplica `pca.transform()` (no `fit_transform()`), garantizando que los datos se proyecten sobre los mismos ejes aprendidos durante el entrenamiento.

**Justificación:** Reduce de 99 dimensiones a 10 componentes principales, preservando la mayor varianza posible. Esto mejora la generalización del modelo y reduce el riesgo de overfitting.

**Formato de salida:**

```
PC1      | PC2      | ... | PC10     | performance
0.04521  | -0.01234 | ... | 0.00891  | 1
-0.03201 | 0.05678  | ... | -0.01045 | 2
...
```

---

### 4. Modelo de Clasificación

**Archivo:** `src/training/train_model.py`

#### 4.1 Modelo Principal: RandomForest

```python
RandomForestClassifier(
    n_estimators=100,          # 100 árboles de decisión
    criterion='gini',          # Índice de Gini para medir impureza
    class_weight='balanced',   # Pondera clases inversamente proporcional a su frecuencia
    random_state=42            # Semilla para reproducibilidad
)
```

**¿Por qué `class_weight='balanced'`?** El etiquetado por percentiles puede generar clases ligeramente desbalanceadas. Este parámetro ajusta automáticamente los pesos para dar más importancia a las clases minoritarias.

#### 4.2 Proceso de Entrenamiento

1. **Carga de datos:** Lee `{exercise}_reduced.csv`
2. **Separación:** `X` = PC1–PC10, `y` = performance
3. **División:** 80% entrenamiento, 20% prueba (`train_test_split(test_size=0.2, random_state=42)`)
4. **Entrenamiento:** `model.fit(X_train, y_train)`
5. **Predicción:** `model.predict(X_test)`
6. **Evaluación:** `accuracy_score`, `classification_report`, `confusion_matrix`
7. **Persistencia:** `joblib.dump(model, 'data/models/{exercise}_model.pkl')`

#### 4.3 Métricas de Evaluación

| Métrica | Descripción | Umbral mínimo (tests) |
|---------|-------------|----------------------|
| Accuracy | Porcentaje de predicciones correctas | ≥ 75% |
| Precision | Proporción de verdaderos positivos entre los predichos como positivos | ≥ 70% |
| Recall | Proporción de verdaderos positivos detectados | — |
| F1-Score | Media armónica de precision y recall | — |

La matriz de confusión se guarda como imagen PNG (300 DPI) en `data/results/confusion_matrices/`.

#### 4.4 Otros Modelos Disponibles

| Modelo | Clase sklearn | Configuración |
|--------|---------------|---------------|
| RandomForest | `RandomForestClassifier` | `n_estimators=100, criterion='gini', class_weight='balanced'` |
| XGBoost* | `GradientBoostingClassifier` | Parámetros por defecto |
| SVM | `SVC` | `kernel='rbf', probability=True` |
| LogisticRegression | `LogisticRegression` | `max_iter=1000` |
| kNN | `KNeighborsClassifier` | `n_neighbors=5` (por defecto) |

*El modelo "XGBoost" en el código usa `GradientBoostingClassifier` de scikit-learn, no la librería XGBoost nativa.

---

### 5. Pipeline de Inferencia (Evaluación)

**Archivos:** `src/evaluation/main_evaluation.py`, `src/evaluation/predict_performance.py`

#### 5.1 Flujo Completo

```
Video MP4 nuevo
    │
    ▼
process_new_video(video_path)
    │  MediaPipe Holistic extrae 33 landmarks × 3 coords por frame
    │  Resultado: DataFrame de N filas × 99 columnas (sin nombres de columna)
    ▼
joblib.load(pca_path)  →  pca.transform(data)
    │  Carga el PCA entrenado y proyecta sobre los mismos ejes
    │  Reduce de 99 a 10 dimensiones
    ▼
joblib.load(model_path)
    │  Carga el modelo entrenado (.pkl)
    ▼
model.predict(data_reduced)
    │  Predicción por frame: array de valores 1, 2 o 3
    ▼
calculate_average_performance(predictions)
    │  Promedia las predicciones numéricas
    ▼
Resultado global:
    promedio ≤ 1.5  →  "High" (Alto rendimiento)
    promedio ≤ 2.5  →  "Moderate" (Rendimiento moderado)
    promedio > 2.5  →  "Low" (Bajo rendimiento)
```

#### 5.2 Diferencias entre Entrenamiento y Evaluación

| Aspecto | Entrenamiento | Evaluación |
|---------|---------------|------------|
| Columnas del DataFrame | Con nombres (`nose_x`, `nose_y`, ...) | Sin nombres (índices 0, 1, 2, ...) |
| PCA | `fit_transform` sobre datos de entrenamiento (aprende ejes + proyecta) | `transform` con el PCA entrenado (solo proyecta sobre los mismos ejes) |
| Etiquetado | Se aplica lógica biomecánica | No se aplica (el modelo ya encapsula ese conocimiento) |
| Salida | Modelo `.pkl` + PCA `.pkl` + métricas | Predicción por frame + promedio global |

---

### 6. Resumen de Criterios de Clasificación

| Ejercicio | Métrica 1 (40%) | Métrica 2 (30%) | Métrica 3 (30%) | Método de Clasificación |
|-----------|-----------------|-----------------|-----------------|------------------------|
| Jump | Altura de tobillos | — | — | Percentiles 33/66 de `avg_ankle_height` |
| Crawl | Coordinación diagonal | Estabilidad de cadera | Fluidez de movimiento | Percentiles 33/67 de `total_score` |
| Sit | Control postural | Simetría corporal | Suavidad de transiciones | Percentiles 33/67 de `total_score` |
| Throw | Secuencia cinemática | Amplitud de movimiento | Estabilidad postural | Percentiles 33/67 de `total_score` |

**Etiquetas:**

| Valor | Significado | Criterio |
|-------|-------------|----------|
| 1 | Alto rendimiento | Top 33% de la puntuación total |
| 2 | Rendimiento moderado | Medio 34% de la puntuación total |
| 3 | Bajo rendimiento | Bottom 33% de la puntuación total |

---

### 7. Diagrama de Flujo Completo del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FASE DE ENTRENAMIENTO                          │
│                                                                     │
│  Videos MP4          process_videos.py         label_data_*.py      │
│  data/raw/{ex}/  ──► MediaPipe Holistic  ──►  Métricas biomec.     │
│                      99 features/frame         + etiquetas 1,2,3    │
│                            │                         │              │
│                            ▼                         ▼              │
│                   {ex}_landmarks.csv        {ex}_labeled.csv        │
│                                                      │              │
│                                              pca_reduction.py       │
│                                              PCA(10 comps)          │
│                                                      │              │
│                                               ┌──────┴──────┐      │
│                                               ▼             ▼      │
│                                        {ex}_pca.pkl  {ex}_reduced  │
│                                                           .csv     │
│                                                      │              │
│                                              train_model.py         │
│                                              RandomForest            │
│                                              80/20 split            │
│                                                      │              │
│                                               ┌──────┴──────┐      │
│                                               ▼             ▼      │
│                                        {ex}_model.pkl   métricas   │
│                                                         + conf.mat.│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     FASE DE EVALUACIÓN                              │
│                                                                     │
│  Video nuevo         predict_performance.py                         │
│  *.mp4          ──►  MediaPipe Holistic  ──►  pca.transform        │
│                      99 features/frame        ({ex}_pca.pkl)        │
│                                                    │                │
│                                               model.predict         │
│                                               ({ex}_model.pkl)      │
│                                                    │                │
│                                                    ▼                │
│                                          Predicción por frame       │
│                                          (1, 2 o 3)                │
│                                                    │                │
│                                          Promedio global            │
│                                          High / Moderate / Low      │
└─────────────────────────────────────────────────────────────────────┘
```
