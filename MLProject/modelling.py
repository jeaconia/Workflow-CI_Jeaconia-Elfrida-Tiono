import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "Abalone_preprocessing"

def main():
    with mlflow.start_run() as run:
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
        X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
        y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
        y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        print(f"Training complete. Accuracy: {acc}")

if __name__ == "__main__":
    main()
