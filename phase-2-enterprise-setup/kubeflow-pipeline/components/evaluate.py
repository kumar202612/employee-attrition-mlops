from kfp import dsl
from kfp.dsl import Artifact, Input, Metrics, Output
from config import BASE_IMAGE


@dsl.component(base_image=BASE_IMAGE)
def evaluate(
    artifact_data: Input[Artifact],
    metrics: Output[Metrics],
):
    with open(artifact_data.path, "r", encoding="utf-8") as f:
        train_output = f.read()

    final_text = f"{train_output} -> used by Evaluate"
    print(f"Evaluate used Train artifact: {final_text}")
    metrics.log_metric("evaluate_message_length", len(final_text))
