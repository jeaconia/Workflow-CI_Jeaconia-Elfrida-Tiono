import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load Data
X_train = pd.read_csv("Abalone_preprocessing/X_train.csv")
X_test = pd.read_csv("Abalone_preprocessing/X_test.csv")
y_train = pd.read_csv("Abalone_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("Abalone_preprocessing/y_test.csv").values.ravel()

#Set Eksperimen
mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("Eksperimen Latih Model Dataset Abalone")

with mlflow.start_run() as run:
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

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model")







