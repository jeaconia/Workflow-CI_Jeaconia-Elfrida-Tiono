import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load Data
X_train = pd.read_csv("WineQT_preprocessing/X_train.csv")
X_test = pd.read_csv("WineQT_preprocessing/X_test.csv")
y_train = pd.read_csv("WineQT_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("WineQT_preprocessing/y_test.csv").values.ravel()

#Set Eksperimen
mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("Eksperimen Latih Model Dataset WineQT")

with mlflow.start_run() as run:
    #Autolog
    mlflow.sklearn.autolog(log_models=False)

    #Training Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
        
    model.fit(X_train, y_train)
    
    #Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    mlflow.sklearn.log_model(model, "model")

    print(f"MLFLOW_ARTIFACT_URI={run.info.artifact_uri}")
    print(f"MLFLOW_RUN_ID={run.info.run_id}")



