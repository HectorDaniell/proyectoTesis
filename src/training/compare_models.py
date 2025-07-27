# compare_models.py
from train_model import train_and_evaluate_model
import pandas as pd

models_to_test = ["RandomForest", "XGBoost", "SVM", "LogisticRegression", "kNN"]
results = []

for model_name in models_to_test:
    print(f"\n=== Training {model_name} ===")
    accuracy, report = train_and_evaluate_model("data/processed/jump_reduced.csv", model_name)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Report": report
    })

# Generate comparative table
results_df = pd.DataFrame(results)
print("\n=== Comparative Results ===")
print(results_df[["Model", "Accuracy"]])

# Save results to CSV
results_df.to_csv("data/results/model_comparison.csv", index=False)
