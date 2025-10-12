# train_model.py - Machine learning model training and evaluation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_confusion_matrix_plot(cm, exercise_name, model_name):
    """
    Create and save a confusion matrix visualization as an image.
    
    Args:
        cm (numpy.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
        exercise_name (str): Name of the exercise
        model_name (str): Name of the model
    """
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bajo Rendimiento', 'Moderado', 'Alto Rendimiento'],
                yticklabels=['Bajo Rendimiento', 'Moderado', 'Alto Rendimiento'],
                ax=ax)
    
    # Customize the plot
    ax.set_title(f'Matriz de Confusión - {model_name}\nEjercicio: {exercise_name.title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = "data/results/confusion_matrices"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the plot
    filename = f"{exercise_name}_{model_name}_confusion_matrix.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Matriz de confusión guardada en: {filepath}")

def train_and_evaluate_model(input_csv, exercise_name, model_name="RandomForest"):
    """
    Train and evaluate a machine learning model for exercise performance classification.
    
    This function implements a complete machine learning pipeline including:
    - Data loading and preprocessing
    - Model selection and training
    - Performance evaluation with multiple metrics
    - Model persistence for later use
    
    Args:
        input_csv (str): Path to the reduced CSV file with PCA components
        exercise_name (str): Name of the exercise
        model_name (str): Name of the model to train (RandomForest, XGBoost, SVM, etc.)
        
    Returns:
        tuple: (accuracy_score, classification_report) for model evaluation
    """
    # Load the reduced data
    df = pd.read_csv(input_csv)
    X = df.drop(['performance'], axis=1)
    y = df['performance']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection based on parameter
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = GradientBoostingClassifier()
    elif model_name == "SVM":
        model = SVC(kernel='rbf', probability=True)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "kNN":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Model not supported")

    # Train the model on training data
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display model performance results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # Create and save confusion matrix visualization
    create_confusion_matrix_plot(cm, exercise_name, model_name)
    
    # Save trained model for later use in evaluation
    model_path = f"data/models/{exercise_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return accuracy, report