import pandas as pd
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
    
    # Separate features (landmarks) from target variable (performance)
    X = df.drop(['performance'], axis=1)
    
    # Apply PCA with 10 components (reduces from ~99 features to 10)
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    
    # Create new DataFrame with PCA components and performance labels
    df_reduced = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(10)])
    df_reduced['performance'] = df['performance']
    
    # Save reduced data to new CSV file
    reduced_csv = input_csv.replace('_labeled.csv', '_reduced.csv')
    df_reduced.to_csv(reduced_csv, index=False)
    print(f"Dimensionality reduction saved in {reduced_csv}")
    
    return reduced_csv