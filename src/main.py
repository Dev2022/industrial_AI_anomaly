import os

from config.config import *

from src.data_generation import generate_dataset
from src.feature_engineering import create_features
from src.anomaly_detection import run_anomaly_detection
from src.failure_prediction import run_failure_prediction
from src.rul_prediction import run_rul_prediction
from src.explainability import generate_shap
from src.optimization import optimize_maintenance

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nGenerating dataset...")
    df = generate_dataset()

    print("\nFeature engineering...")
    df = create_features(df)

    print("\nAnomaly detection...")
    df, anomaly_features = run_anomaly_detection(df)

    print("\nFailure prediction...")
    feature_cols = anomaly_features + ["anomaly_label"]

    df, model, X_test, metrics = run_failure_prediction(
        df,
        feature_cols
    )

    print("\nRUL prediction...")
    X = df[feature_cols]

    df, rul_model = run_rul_prediction(df, X)

    print("\nGenerating SHAP explainability...")
    generate_shap(model, X_test)

    print("\nOptimization...")
    results_df, prob = optimize_maintenance(df)

    results_df.to_csv(
        "results/maintenance_plan.csv",
        index=False
    )

    df.to_csv(
        "results/final_dataset.csv",
        index=False
    )

    print("\n====================================")
    print("FINAL METRICS")
    print("====================================")

    for k, v in metrics.items():

        print(f"{k}: {round(v, 4)}")

    print("\nDONE.")

if __name__ == "__main__":

    main()