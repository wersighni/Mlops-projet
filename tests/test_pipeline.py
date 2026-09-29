"""Tests unitaires du pipeline ML."""

from model_pipeline import prepare_data, train_model


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert "Exited" not in X_train.columns


def test_train_model():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train, n_estimators=10)
    assert model.score(X_test, y_test) > 0.7

