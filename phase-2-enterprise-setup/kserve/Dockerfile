FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "mlserver==1.7.1" \
    "mlserver-mlflow==1.7.1" \
    "mlflow==3.14.0" \
    "scikit-learn==1.9.0" \
    "pandas==2.3.3" \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "skops==0.14.0" \
    "uvloop==0.21.0"

RUN useradd --create-home --uid 1000 mlserver
WORKDIR /home/mlserver
USER mlserver

ENV MLSERVER_MODELS_DIR=/mnt/models
ENV MLSERVER_MODEL_IMPLEMENTATION=mlserver_mlflow.MLflowRuntime
ENV MLSERVER_HTTP_PORT=8080
ENV MLSERVER_GRPC_PORT=8081

EXPOSE 8080 8081 8082

CMD ["mlserver", "start", "/mnt/models"]
