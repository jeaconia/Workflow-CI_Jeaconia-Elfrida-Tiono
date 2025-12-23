import os
import mlflow
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "Abalone_preprocessing"
MODEL_DIR = "model_artifact"

def main():
    mlflow.autolog(log_models=False)

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")

    mlflow.log_artifacts(MODEL_DIR, artifact_path="model")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy_manual", accuracy)

if __name__ == "__main__":
    main()
