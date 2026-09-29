"""
deploiement_prefect.py - Deploiement et planification des flows Prefect.
Lance un "worker" qui execute les flows a la demande ou selon un planning.
Usage : python deploiement_prefect.py
"""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, evaluate_flow, predict_flow, train_flow

if __name__ == "__main__":
    # Pipeline complet : planifie tous les jours a 02h00
    all_deploy = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["full-pipeline", "mlops"],
    )
    # Entrainement : planifie tous les jours a 03h00
    train_deploy = train_flow.to_deployment(
        name="ml-pipeline-train",
        cron="0 3 * * *",
        tags=["training", "mlops"],
    )
    # Flows lances manuellement
    evaluate_deploy = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
    )
    code_deploy = code_flow.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"])
    predict_deploy = predict_flow.to_deployment(
        name="ml-pipeline-predict", tags=["prediction", "mlops"]
    )

    serve(all_deploy, train_deploy, evaluate_deploy, code_deploy, predict_deploy)
