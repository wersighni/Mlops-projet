"""
main.py - Execution du pipeline ML via des arguments CLI.
Exemples : python main.py --prepare | --train | --evaluate | --all
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Prediction du churn client"
    )
    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Chemin du fichier CSV"
    )
    parser.add_argument(
        "--model", default="classifier.joblib", help="Chemin du fichier modele"
    )
    parser.add_argument("--prepare", action="store_true", help="Preparer les donnees")
    parser.add_argument(
        "--train", action="store_true", help="Entrainer et sauvegarder le modele"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Charger et evaluer le modele"
    )
    parser.add_argument("--all", action="store_true", help="Executer tout le pipeline")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.evaluate or args.all):
        parser.print_help()
        return

    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.train or args.all:
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    if args.evaluate or args.all:
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
