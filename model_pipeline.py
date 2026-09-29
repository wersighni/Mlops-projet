"""
model_pipeline.py
Fonctions modulaires du pipeline ML de prediction du churn client.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(file_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge et pretraite les donnees. Retourne X_train, X_test, y_train, y_test."""
    df = pd.read_csv(file_path)
    df = df.drop_duplicates().dropna()
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])
    df = df.drop(columns=COLUMNS_TO_DROP)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("[prepare_data] Train :", X_train.shape, "| Test :", X_test.shape)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraine un RandomForestClassifier et retourne le modele."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("[train_model] Modele entraine avec succes.")
    return model


def evaluate_model(model, X_test, y_test):
    """Evalue le modele : accuracy, matrice de confusion, rapport."""

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print("[evaluate_model] Accuracy :", round(accuracy * 100, 2), "%")
    print("Matrice de confusion :")
    print(matrix)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    return {"accuracy": accuracy, "confusion_matrix": matrix}


def save_model(model, file_path="classifier.joblib"):
    """Sauvegarde le modele avec joblib."""
    joblib.dump(model, file_path)
    print("[save_model] Modele sauvegarde dans :", file_path)


def load_model(file_path="classifier.joblib"):
    """Charge un modele sauvegarde avec joblib."""
    model = joblib.load(file_path)
    print("[load_model] Modele charge depuis :", file_path)
    return model
