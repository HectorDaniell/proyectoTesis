import pandas as pd
import joblib
from sklearn.decomposition import PCA

# Function to apply Principal Component Analysis for dimensionality reduction
def apply_pca(input_csv):
    """
    Apply PCA dimensionality reduction to labeled landmark data.
    
    This function reduces the high-dimensional landmark data to a lower-dimensional
    representation while preserving the most important variance in the data.
    This is crucial for improving model performance and reducing computational complexity.
    
    Args:
        input_csv (str): Path to the labeled CSV file containing landmark data
        
    Returns:
        str: Path to the reduced CSV file with PCA components
    """
    # Load the labeled data
    df = pd.read_csv(input_csv)
    
    # Select only the 99 landmark columns (those ending in _x, _y, _z),
    # excluding intermediate scoring columns added by the labeling step.
    # This ensures PCA is fitted on the same features available during evaluation.
    landmark_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]
    X = df[landmark_cols]
    
    # Apply PCA retaining 95% of explained variance (component count varies per exercise)
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)

    n_components = X_reduced.shape[1]
    print(f"PCA selected {n_components} components to explain 95% of variance")

    # Create new DataFrame with PCA components and performance labels
    df_reduced = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])
    df_reduced['performance'] = df['performance']
    
    # Save reduced data to new CSV file
    reduced_csv = input_csv.replace('_labeled.csv', '_reduced.csv')
    df_reduced.to_csv(reduced_csv, index=False)
    print(f"Dimensionality reduction saved in {reduced_csv}")
    
    # Persist the fitted PCA object so evaluation uses the same transformation
    pca_path = input_csv.replace('_labeled.csv', '_pca.pkl')
    joblib.dump(pca, pca_path)
    print(f"PCA object saved in {pca_path}")
    
    return reduced_csv