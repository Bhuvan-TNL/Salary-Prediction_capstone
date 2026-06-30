import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import os

print("🚀 Explainability script started...")

# 1) Load model and test data
model = joblib.load("models/random_forest_model.joblib")
X_test = pd.read_csv("notebook/X_test.csv")
print("✅ Model and test data loaded:", X_test.shape)

# 2) Feature Importance (Random Forest built-in)
importances = model.feature_importances_
indices = importances.argsort()[::-1][:15]  # top 15 features

plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), X_test.columns[indices], rotation=90)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()

os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/feature_importance.png")
plt.close()
print("✅ Feature importance saved to screenshots/feature_importance.png")

# 3) SHAP Values (for detailed interpretability)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:200])  # use 200 samples for speed

plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test.iloc[:200], show=False)
plt.tight_layout()
plt.savefig("screenshots/shap_summary.png", bbox_inches="tight")
plt.close()
print("✅ SHAP summary plot saved to screenshots/shap_summary.png")
