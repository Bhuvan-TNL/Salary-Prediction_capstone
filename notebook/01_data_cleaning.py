import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import train_test_split

print("🚀 Script started...")

# 1) Load dataset
df = pd.read_csv("data/expected_ctc.csv")
print("Original shape:", df.shape)

# 2) Drop ID columns
for col in ['IDX', 'Applicant_ID']:
    if col in df.columns:
        df = df.drop(columns=[col])

# 3) Drop columns with >50% missing
thresh = int(0.5 * len(df))
df = df.dropna(axis=1, thresh=thresh)
print("After dropping sparse cols:", df.shape)

# 4) Target column
target_col = 'Expected_CTC'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# 5) Feature engineering: exp_ratio
if 'Total_Experience' in df.columns and 'Total_Experience_in_field_applied' in df.columns:
    df['exp_ratio'] = (df['Total_Experience_in_field_applied'] / (df['Total_Experience'] + 1)).clip(0, 1)

# 6) Split numeric vs categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# 7) Handle missing values
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna("Missing").astype(str)

# 8) Frequency encode high-cardinality categories (>30 unique)
freq_maps = {}
new_cols = []
for c in cat_cols:
    if df[c].nunique() > 30:
        freq = df[c].value_counts(normalize=True)
        new_col = c + "_freq"
        df[new_col] = df[c].map(freq).fillna(0.0)
        freq_maps[c] = freq.to_dict()
        new_cols.append(new_col)
    else:
        new_cols.append(c)

# 9) One-hot encode low-cardinality
df = pd.get_dummies(df, columns=[c for c in new_cols if c in cat_cols and df[c].nunique() <= 30], drop_first=True)

# 10) Prepare features and target
X = df.drop(columns=[target_col])
y = np.log1p(df[target_col].astype(float))  # log transform

# 11) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 12) Save outputs
os.makedirs("notebook", exist_ok=True)
os.makedirs("models", exist_ok=True)

X_train.to_csv("notebook/X_train.csv", index=False)
X_test.to_csv("notebook/X_test.csv", index=False)
y_train.to_csv("notebook/y_train.csv", index=False)
y_test.to_csv("notebook/y_test.csv", index=False)

joblib.dump(freq_maps, "models/freq_maps.joblib")

print("✅ Data cleaned, split, and saved.")
