from kfp import dsl
from kfp.dsl import Artifact, Metrics, Output
from config import BASE_IMAGE


@dsl.component(base_image=BASE_IMAGE)
def train(
    input_message: str,
    artifact_data: Output[Artifact],
    metrics: Output[Metrics],
):
    print(f"Train used Preprocess output: {input_message}")
    artifact_text = f"{input_message} -> output from Train"

    with open(artifact_data.path, "w", encoding="utf-8") as f:
        f.write(artifact_text)

    metrics.log_metric("train_message_length", len(artifact_text))
