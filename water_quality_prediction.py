import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv("water_potability.csv")

print(data.head())
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
sns.countplot(x='Potability', data=data)
plt.title("Potable vs Non-Potable Water")
plt.show()
# ===============================
# Separate Features and Target
# ===============================
X = data.drop("Potability", axis=1)
y = data["Potability"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Model training completed!")
y_pred = model.predict(X_test)

print("Predictions completed!")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ===============================
# Feature Importance
# ===============================

import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance in Water Quality Prediction")
plt.gca().invert_yaxis()
plt.show()
import joblib

joblib.dump(model, "water_quality_model.pkl")

print("Model saved successfully!")
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred))
y_pred = best_model.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import roc_curve, auc

y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (AUC = %0.2f)" % roc_auc)
plt.show()
