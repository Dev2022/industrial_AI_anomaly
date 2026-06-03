import shap
import matplotlib.pyplot as plt

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

def generate_shap(model, X_test):

    explainer = shap.TreeExplainer(model)

    sample = X_test.sample(500, random_state=42)

    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(12, 6))

    shap.summary_plot(
        shap_values,
        sample,
        show=False
    )

    plt.savefig(
        "results/shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()