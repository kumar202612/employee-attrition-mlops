from kfp import dsl
from kfp.dsl import Metrics, Output
from config import BASE_IMAGE


@dsl.component(base_image=BASE_IMAGE)
def preprocess(
    message: str,
    metrics: Output[Metrics],
) -> str:
    print(f"Preprocess received: {message}")
    output_message = f"{message} -> output from Preprocess"
    metrics.log_metric("preprocess_message_length", len(output_message))
    return output_message
