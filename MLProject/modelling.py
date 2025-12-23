import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "Abalone_preprocessing"

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def main():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SklearnModelWrapper(model),
        input_example=X_train.iloc[:5]
    )

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
