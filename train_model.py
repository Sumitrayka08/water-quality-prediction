import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("water_potability.csv")  # Change to your dataset file

# Features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 Create Pipeline (Scaler + Random Forest)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Save ONLY ONE FILE
joblib.dump(pipeline, "water_quality_model.pkl")

print("✅ Model trained and saved successfully.")
