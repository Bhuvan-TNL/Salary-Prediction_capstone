import pandas as pd

print("Trying to read CSV...")
df = pd.read_csv("data/expected_ctc.csv")
print("✅ CSV loaded successfully!")
print("Shape:", df.shape)
print(df.head())
