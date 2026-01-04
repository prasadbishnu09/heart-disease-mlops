import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_experiment("Heart Disease Classification")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load data
df = pd.read_csv("data/processed/heart_clean.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------
# Logistic Regression
# --------------------
with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    mlflow.log_metric("accuracy", lr_acc)
    mlflow.log_metric("roc_auc", lr_auc)

    mlflow.sklearn.log_model(lr, "logistic_model")

    print("Logistic Regression Accuracy:", lr_acc)
    print("Logistic Regression ROC-AUC:", lr_auc)

# --------------------
# Random Forest
# --------------------
with mlflow.start_run(run_name="Random_Forest"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    mlflow.log_metric("accuracy", rf_acc)
    mlflow.log_metric("roc_auc", rf_auc)

    mlflow.sklearn.log_model(rf, "random_forest_model")

    print("Random Forest Accuracy:", rf_acc)
    print("Random Forest ROC-AUC:", rf_auc)

# --------------------
# Save best model (Random Forest)
# --------------------
joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully")