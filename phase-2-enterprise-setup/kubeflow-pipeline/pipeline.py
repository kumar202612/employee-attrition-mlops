import urllib3
import warnings

from kfp import dsl
from kfp.client import Client

from config import KFP_ENDPOINT, KFP_UI_ENDPOINT
from components.evaluate import evaluate
from components.preprocess import preprocess
from components.train import train

warnings.filterwarnings("ignore", category=FutureWarning, module="kfp")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dsl.pipeline(name="ml-pipeline", description="Simple pipeline showing parameter and artifact passing")
def ml_pipeline(
    message: str = "hello from pipeline",
):
    preprocess_task = preprocess(message=message)
    preprocess_task.set_display_name("Preprocess")
    preprocess_task.set_retry(num_retries=2, backoff_duration="30s")

    train_task = train(
        input_message=preprocess_task.outputs["Output"],
    )
    train_task.set_display_name("Train")
    train_task.set_retry(num_retries=2, backoff_duration="30s")

    evaluate_task = evaluate(
        artifact_data=train_task.outputs["artifact_data"],
    )
    evaluate_task.set_display_name("Evaluate")
    evaluate_task.set_retry(num_retries=2, backoff_duration="30s")


if __name__ == "__main__":
    client = Client(host=KFP_ENDPOINT, ui_host=KFP_UI_ENDPOINT, verify_ssl=False)
    run = client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={
            "message": "hello from pipeline",
        },
    )
    print(f"Run: {run.run_id}")
    print(f"URL: {KFP_UI_ENDPOINT}/#/runs/details/{run.run_id}")
