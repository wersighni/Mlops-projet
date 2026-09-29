"""
pipeline_prefect.py - Orchestration du pipeline ML avec Prefect.
Usage : python pipeline_prefect.py --flow all|train|evaluate|code|predict
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from main import evaluate_model, load_model, prepare_data, save_model, train_model

DATA = "Churn_Modelling.csv"
MODEL = "classifier.joblib"
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd, check=True):
    """Execute une commande et affiche son resultat."""
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, check=False)  # nosec B603
    if check and result.returncode != 0:
        raise RuntimeError("Echec de la commande : " + " ".join(cmd))
    return result.returncode


# ---------- Taches liees au code ----------
@task(name="git_pull")
def git_pull():
    """Recupere la derniere version du projet depuis GitHub."""
    run(["git", "pull", "origin", "main"], check=False)
@task(name="install_dependencies")
def install_dependencies():
    """Installe les dependances du projet."""
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])


@task(name="format_code")
def format_code():
    """Formate le code avec black."""
    run([sys.executable, "-m", "black", *FILES])


@task(name="check_quality")
def check_quality():
    """Verifie la qualite du code avec flake8."""
    run([sys.executable, "-m", "flake8", "--max-line-length=100", *FILES], check=False)


@task(name="check_security")
def check_security():
    """Analyse la securite du code avec bandit."""
    run([sys.executable, "-m", "bandit", "-q", *FILES], check=False)


@task(name="run_tests")
def run_tests():
    """Execute les tests unitaires avec pytest."""
    run([sys.executable, "-m", "pytest", "-q", "tests/"])


# ---------- Taches liees aux donnees et au modele ----------
@task(name="prepare_data", cache_policy=NO_CACHE)
def prepare_task():
    """Prepare les donnees."""
    return prepare_data(DATA)


@task(name="train_model", cache_policy=NO_CACHE)
def train_task(X_train, y_train):
    """Entraine le modele."""
    return train_model(X_train, y_train)


@task(name="save_model", cache_policy=NO_CACHE)
def save_task(model):
    """Sauvegarde le modele."""
    save_model(model, MODEL)


@task(name="load_model", cache_policy=NO_CACHE)
def load_task():
    """Charge le modele."""
    return load_model(MODEL)


@task(name="evaluate_model", cache_policy=NO_CACHE)
def evaluate_task(model, X_test, y_test):
    """Evalue le modele."""
    return evaluate_model(model, X_test, y_test)


@task(name="predict", cache_policy=NO_CACHE)
def predict_task(model, X_test):
    """Predit le churn de 5 clients."""
    predictions = model.predict(X_test.head(5))
    print("[predict] Predictions sur 5 clients :", list(predictions))
    return predictions


# ---------- Flows ----------
@flow(name="code")
def code_flow():
    """Formatage, qualite, securite et tests."""
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow():
    """Preparation des donnees et entrainement."""
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    """Chargement et evaluation du modele."""
    X_train, X_test, y_train, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, X_test, y_test)


@flow(name="predict")
def predict_flow():
    """Prediction avec le modele sauvegarde."""
    X_train, X_test, y_train, y_test = prepare_task()
    predict_task(load_task(), X_test)


@flow(name="all")
def all_flow():
    """Pipeline complet."""
    git_pull()
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    evaluate_task(model, X_test, y_test)
    predict_task(model, X_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "predict": predict_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
