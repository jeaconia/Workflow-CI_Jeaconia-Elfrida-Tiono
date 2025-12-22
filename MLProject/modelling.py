import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load Data
X_train = pd.read_csv("Github/data/processed/X_train.csv")
X_test = pd.read_csv("Github/data/processed/X_test.csv")
y_train = pd.read_csv("Github/data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("Github/data/processed/y_test.csv").values.ravel()

#Set Tracking UI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#Set Eksperimen
mlflow.set_experiment("Eksperimen Latih Model Dataset WineQT")

#Autolog
mlflow.sklearn.autolog()

#Training Model
with mlflow.start_run():
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
