import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Función para entrenar el modelo y guardarlo
def train_and_save_model(input_csv):
    df = pd.read_csv(input_csv)
    
    X = df.drop(['performance'], axis=1)
    y = df['performance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    

        # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    models_folder = '/Users/danieloviedo/Library/CloudStorage/OneDrive-UniversidadCatólicadeSantaMaría/UCSM/10mo Semestre/INVESTIGACIÓN II/Proyecto Tesis/proyectoTesis/data/models/'
    
    # Definir el nombre del archivo modelo
    model_filename = os.path.join(models_folder, os.path.basename(input_csv).replace('_reduced.csv', '_model.pkl'))
    joblib.dump(clf, model_filename)
    print(f"Modelo guardado en {model_filename}")