import pandas as pd
from sklearn.decomposition import PCA

# Función para aplicar PCA
def apply_pca(input_csv):
    df = pd.read_csv(input_csv)
    
    X = df.drop(['performance'], axis=1)
    
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    
    df_reduced = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(10)])
    df_reduced['performance'] = df['performance']
    
    reduced_csv = input_csv.replace('_labeled.csv', '_reduced.csv')
    df_reduced.to_csv(reduced_csv, index=False)
    print(f"Dimensionalidad reducida guardada en {reduced_csv}")
    
    return reduced_csv