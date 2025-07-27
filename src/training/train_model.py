# train_model.py - Machine learning model training and evaluation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

def train_and_evaluate_model(input_csv, model_name="RandomForest"):
    """
    Train and evaluate a machine learning model for exercise performance classification.
    
    This function implements a complete machine learning pipeline including:
    - Data loading and preprocessing
    - Model selection and training
    - Performance evaluation with multiple metrics
    - Model persistence for later use
    
    Args:
        input_csv (str): Path to the reduced CSV file with PCA components
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
    
    # Display model performance results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # Save trained model for later use in evaluation
    model_path = f"data/models/{model_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return accuracy, report