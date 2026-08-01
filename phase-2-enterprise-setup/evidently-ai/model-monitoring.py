import argparse
import random
from datetime import datetime, timedelta

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import (
    bar_plot_panel,
    counter_panel,
    line_plot_panel,
    pie_plot_panel,
    text_panel,
)
from evidently.ui.workspace import RemoteWorkspace

EVIDENTLY_URL = "http://localhost:8000" 
EVIDENTLY_SECRET = None

PROJECT_ID = "019fbb5d-b414-700b-8840-0a55ba10f28f"


FEATURES = [f"feature_{i}" for i in range(8)]
TARGET = "target"
PRED_LABEL = "prediction"
PRED_PROBA = "prediction_proba"

def train_model_and_reference():
    """Return (model, reference_Dataset, production_DataFrame, data_definition)."""
    X, y = make_classification(
        n_samples=4000,
        n_features=len(FEATURES),
        n_informative=5,
        weights=[0.7, 0.3],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y

    train_df, prod_df = train_test_split(df, test_size=0.6, random_state=42)

    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(train_df[FEATURES], train_df[TARGET])

    reference_df = train_df.copy()
    reference_df[PRED_LABEL] = model.predict(reference_df[FEATURES])
    reference_df[PRED_PROBA] = model.predict_proba(reference_df[FEATURES])[:, 1]

    data_definition = DataDefinition(
        numerical_columns=FEATURES,
        classification=[
            BinaryClassification(
                target=TARGET,
                prediction_labels=PRED_LABEL,
                prediction_probas=PRED_PROBA,
                pos_label=1,
            )
        ],
    )

    reference_ds = Dataset.from_pandas(reference_df, data_definition=data_definition)
    return model, reference_ds, prod_df, data_definition

def build_dashboard(project):
    dash = project.dashboard

    # Idempotency guard: if panels already exist, do nothing (no duplicates).
    if len(dash.model().panels) > 0:
        print("Dashboard already has panels — skipping build.")
        return

    dash.add_panel(
        text_panel(title="Model Health Overview"),
    )
    dash.add_panel(
        counter_panel(
            title="Total predictions checked",
            size="half",
            values=[PanelMetric(legend="predictions", metric="RowCount")],
            aggregation="sum",
        ),
    )
    dash.add_panel(
        counter_panel(
            title="Accuracy",
            size="half",
            values=[PanelMetric(legend="Accuracy", metric="Accuracy")],
            aggregation="last",
        ),
    )
    dash.add_panel(
        counter_panel(
            title="ROC AUC — higher is better",
            size="half",
            values=[PanelMetric(legend="ROC AUC", metric="RocAuc")],
            aggregation="last",
        ),
    )
    dash.add_panel(
        counter_panel(
            title="Data Drift share — lower is better",
            size="half",
            values=[
                PanelMetric(
                    legend="drift share",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                )
            ],
            aggregation="last",
        ),
    )

    dash.add_panel(text_panel(title="How good are the predictions?"))
    dash.add_panel(
        line_plot_panel(
            title="Prediction quality over time (higher is better)",
            values=[
                PanelMetric(legend="Accuracy", metric="Accuracy"),
                PanelMetric(legend="Precision", metric="Precision"),
                PanelMetric(legend="Recall", metric="Recall"),
                PanelMetric(legend="Overall balance F1 score", metric="F1Score"),
            ],
            size="full",
        ),
    )
    dash.add_panel(
        line_plot_panel(
            title="Class separation over time (ROC AUC — higher is better)",
            size="half",
            values=[PanelMetric(legend="ROC AUC", metric="RocAuc")],
        ),
    )
    dash.add_panel(
        line_plot_panel(
            title="Log Loss — lower is better",
            size="half",
            values=[PanelMetric(legend="Log Loss", metric="LogLoss")],
        ),
    )
    dash.add_panel(
        bar_plot_panel(
            title="Each class: when flagged, how often right (Precision by class — higher is better)",
            size="half",
            values=[PanelMetric(legend="class {{label}}", metric="PrecisionByLabel")],
        ),
    )
    dash.add_panel(
        bar_plot_panel(
            title="how many real cases caught (Recall by class — higher is better)",
            size="half",
            values=[PanelMetric(legend="class {{label}}", metric="RecallByLabel")],
        ),
    )

    dash.add_panel(text_panel(title="Is the incoming data normal?"))
    dash.add_panel(
        line_plot_panel(
            title="Drifted Columns Count — lower is better",
            values=[
                PanelMetric(
                    legend="drifted columns",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                )
            ],
            size="full",
        ),
    )
    dash.add_panel(
        line_plot_panel(
            title="Drift Score per feature — lower is better",
            values=[
                PanelMetric(legend=c, metric="ValueDrift", metric_labels={"column": c})
                for c in FEATURES[:4]
            ],
            size="full",
        ),
    )
    dash.add_panel(
        bar_plot_panel(
            title="Missing or empty values — lower is better)",
            size="half",
            values=[PanelMetric(legend="missing values", metric="DatasetMissingValueCount")],
        ),
    )
    dash.add_panel(
        pie_plot_panel(
            title="Mix of outcomes (Class Balance — neither good nor bad)",
            size="half",
            values=[
                PanelMetric(
                    legend="{{label}}",
                    metric="CategoryCount",
                    metric_labels={"column": TARGET},
                )
            ],
            aggregation="last",
        ),
    )

    project.save()
    print("Dashboard built.")


def add_precision_recall_panels(project):
    """Add standalone Precision / Recall / F1 counters, without touching existing panels."""
    dash = project.dashboard

    existing_titles = {p.title for p in dash.model().panels}
    new_panels = [
        (
            "Precision — higher is better",
            "Precision",
        ),
        (
            "Recall — higher is better",
            "Recall",
        ),
        (
            "F1 score — higher is better",
            "F1Score",
        ),
    ]

    added_any = False
    for title, metric in new_panels:
        if title in existing_titles:
            print(f"Panel '{title}' already exists — skipping.")
            continue
        dash.add_panel(
            counter_panel(
                title=title,
                size="half",
                values=[PanelMetric(legend=metric, metric=metric)],
                aggregation="last",
            ),
        )
        added_any = True

    if added_any:
        project.save()
        print("Precision/Recall/F1 panels added.")
    else:
        print("Precision/Recall/F1 panels already present — nothing to do.")


def make_batch(prod_df, model, drift_strength, seed):
    """Sample a production batch, optionally drift it, and score it."""
    batch = prod_df.sample(n=300, random_state=seed).copy()
    for col in FEATURES[:3]:
        batch[col] = batch[col] + drift_strength
    batch[PRED_LABEL] = model.predict(batch[FEATURES])
    batch[PRED_PROBA] = model.predict_proba(batch[FEATURES])[:, 1]
    return batch


def run_report(batch, reference_ds, data_definition, timestamp):
    current_ds = Dataset.from_pandas(batch, data_definition=data_definition)
    report = Report(
        [
            ClassificationPreset(),
            DataDriftPreset(),
            DataSummaryPreset(),
        ]
    )
    return report.run(current_ds, reference_ds, timestamp=timestamp)


def push_snapshot(ws, project, snapshot, label):
    """Push one snapshot. Errors are caught so a monitoring run never crashes."""
    try:

        ws.add_run(project.id, snapshot, include_data=True)
        print(f"  pushed snapshot @ {label}")
        return True
    except Exception as exc:
        print(f"  ! failed to push snapshot @ {label}: {exc}")
        return False


def monitoring_tick(ws, project, model, reference_ds, prod_df, data_definition):
    """One real monitoring run: push the LATEST batch at the current time.

    Run this on a schedule (cron / Kubernetes CronJob). Each call adds one new
    data point, so the dashboard updates on its own — no flags to toggle.
    """
    drift_now = random.uniform(0.0, 2.5)
    seed = random.randint(0, 10**6)
    now = datetime.now()
    snapshot = run_report(
        make_batch(prod_df, model, drift_now, seed), reference_ds, data_definition, now
    )
    return push_snapshot(ws, project, snapshot, now.strftime("%Y-%m-%d %H:%M:%S"))


def backfill_history(ws, project, model, reference_ds, prod_df, data_definition, n_days):
    """One-time seed: push n_days of past snapshots so a new dashboard has a trend."""
    print(f"Backfilling {n_days} day(s) of history ...")
    for day in range(n_days):
        ts = datetime.now() - timedelta(days=(n_days - 1 - day))
        drift = day * 0.35
        snapshot = run_report(
            make_batch(prod_df, model, drift, seed=day), reference_ds, data_definition, ts
        )
        push_snapshot(ws, project, snapshot, ts.strftime("%Y-%m-%d"))


def main():
    args = parse_args()

    print("Connecting to", EVIDENTLY_URL)
    ws = RemoteWorkspace(EVIDENTLY_URL, secret=EVIDENTLY_SECRET)

    project = ws.get_project(PROJECT_ID)
    if project is None:
        raise SystemExit(
            f"No project found with id {PROJECT_ID!r}. "
            f"Copy the exact UUID from the project page in the UI."
        )
    print("Using project:", project.id)

    build_dashboard(project)
    add_precision_recall_panels(project)

    print("Training model and reference data ...")
    model, reference_ds, prod_df, data_definition = train_model_and_reference()

    if args.backfill > 0:
        try:
            ws.add_dataset(
                project.id,
                reference_ds,
                name="Reference data (baseline)",
                description="Baseline the live data is compared against.",
            )
            print("Uploaded reference dataset.")
        except Exception as exc:
            print("  ! could not upload reference dataset:", exc)

        backfill_history(
            ws, project, model, reference_ds, prod_df, data_definition, args.backfill
        )

    print("Pushing current monitoring tick ...")
    monitoring_tick(ws, project, model, reference_ds, prod_df, data_definition)

    print("\nDone. Open the project -> Dashboard tab (refresh if needed). "
          "Run again — or schedule it — to keep adding points.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evidently monitoring run.")
    parser.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="One-time: also push N days of past snapshots to seed the dashboard "
             "(e.g. --backfill 7). Default 0 = just push the current point.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()