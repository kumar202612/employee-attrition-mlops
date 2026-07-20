import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MLFLOW_TRACKING_URI = "http://52.13.161.160:32287/"
EXPERIMENT_NAME = "employee-attrition"
DATA_PATH = "datasets/employee_attrition.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "gradient_boosting"
estimator = GradientBoostingClassifier(random_state=42)


@mlflow.trace(name="load_dataset", span_type="DATASET")
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Employee ID"])

    int_cols = df.select_dtypes(include="int64").columns.tolist()
    df[int_cols] = df[int_cols].astype("float64")

    train_df = df[df["dataset_type"] == "train"].drop(columns=["dataset_type"])
    test_df = df[df["dataset_type"] == "test"].drop(columns=["dataset_type"])

    X_train = train_df.drop(columns=["Attrition"])
    y_train = (train_df["Attrition"] == "Left").astype(int)

    X_test = test_df.drop(columns=["Attrition"])
    y_test = (test_df["Attrition"] == "Left").astype(int)

    return X_train, X_test, y_train, y_test


@mlflow.trace(name="build_pipeline")
def build_pipeline(estimator, X_train):
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


@mlflow.trace(name="train_model")
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


@mlflow.trace(name="evaluate_model")
def evaluate_model(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


X_train, X_test, y_train, y_test = load_dataset()

description = f"{type(estimator).__name__} trained on the employee attrition dataset."

with mlflow.start_run(run_name=MODEL_NAME, description=description) as run:

    pipeline = build_pipeline(estimator, X_train)
    pipeline = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)

    mlflow.log_params(
        {
            "candidate": MODEL_NAME,
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "n_features": X_train.shape[1],
            **{f"model__{k}": v for k, v in estimator.get_params().items()},
        }
    )

    mlflow.log_metrics(metrics)

    mlflow.set_tags(
        {
            "model_type": type(estimator).__name__,
            "dataset": "employee_attrition",
        }
    )

    signature = infer_signature(X_train, pipeline.predict(X_train))

    skops_trusted_types = skops.io.get_untrusted_types(data=skops.io.dumps(pipeline))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="sklearn-model",
        signature=signature,
        input_example=X_train.head(5),
        skops_trusted_types=skops_trusted_types,
    )

    metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"[{MODEL_NAME}] run_id={run.info.run_id} {metrics_str}")
