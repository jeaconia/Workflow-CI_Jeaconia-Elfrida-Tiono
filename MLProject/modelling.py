import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "Abalone_preprocessing"


def main():
    mlflow.set_experiment("abalone-experiment")

        print(f"MLFLOW_RUN_ID={run.info.run_id}")

        # Load data
        X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
        X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
        y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
        y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

        # Model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )

        # Evaluation
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        print(f"Training selesai | ACC={acc:.4f}")


if __name__ == "__main__":
    main()

