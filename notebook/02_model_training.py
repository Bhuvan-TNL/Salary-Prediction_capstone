import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("🚀 Script started...")

# Load data
X_train = pd.read_csv("../notebook/X_train.csv")
X_test = pd.read_csv("../notebook/X_test.csv")
y_train = pd.read_csv("../notebook/y_train.csv")
y_test = pd.read_csv("../notebook/y_test.csv")

print(f"✅ Data loaded: {X_train.shape} {X_test.shape}")

# Train Random Forest
print("⏳ Training started...")
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train.values.ravel())
print("✅ Training finished")

# Predictions
y_pred = model.predict(X_test)
print("✅ Predictions done")

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)
print(f"📊 RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save trained model
with open("../models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved to ../models/random_forest.pkl")
print("✅ Script finished.")