import sys

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "http://52.13.161.160:32287/"
MODEL_NAME = "employee-attrition-classifier"
CHAMPION_ALIAS = "champion"
RUN_ID = "8a557a5d7bf844febac4ec391b9777fb"  # set this to the run_id to register and promote

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def register_and_promote(run_id):
    client = MlflowClient()

    candidate_run = client.get_run(run_id)

    model_outputs = candidate_run.outputs.model_outputs if candidate_run.outputs else []
    if not model_outputs:
        raise ValueError(f"Run {run_id} has no logged model.")

    model_uri = f"models:/{model_outputs[0].model_id}"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.update_model_version(
        name=MODEL_NAME,
        version=registered_model.version,
        description=f"Promoted from run {run_id} ({candidate_run.info.run_name}).",
    )
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=registered_model.version,
        key="validation_status",
        value="approved",
    )
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=registered_model.version,
    )

    print(f"Promoted : v{registered_model.version} is the new champion (from run {run_id})")


if __name__ == "__main__":
    if RUN_ID == "<RUN_ID>":
        print("Set RUN_ID in register_model.py before running.")
        sys.exit(1)

    register_and_promote(RUN_ID)
