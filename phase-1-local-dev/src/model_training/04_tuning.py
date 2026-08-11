import json
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config.paths import PREPROCESSED_TRAIN_PATH, MODEL_PATH, ARTIFACT_DIR

CV_RESULTS_PATH = ARTIFACT_DIR / "cv_results.json"

# Must match the CANDIDATE_MODELS keys/instances used in 03_cross_validation.py
CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

# Which metric decides the winner. Recall is prioritized by default since
# missing an at-risk employee (false negative) is usually costlier than
# flagging a safe one (false positive) in attrition prediction.
SELECTION_METRIC = "recall"


def build_pipeline(preprocessing_steps, classifier):
    return Pipeline(
        steps=[(name, clone(step)) for name, step in preprocessing_steps]
        + [("classifier", classifier)]
    )


def select_best_model():
    if not CV_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{CV_RESULTS_PATH} not found — run 03_cross_validation.py first."
        )

    with open(CV_RESULTS_PATH) as f:
        cv_results = json.load(f)

    best = max(cv_results, key=lambda r: r[SELECTION_METRIC])
    print(f"Best algorithm by {SELECTION_METRIC}: {best['algorithm']} "
          f"({SELECTION_METRIC}={best[SELECTION_METRIC]:.4f})")
    print("Full comparison:")
    for r in sorted(cv_results, key=lambda r: r[SELECTION_METRIC], reverse=True):
        print(f"  {r['algorithm']:<20} accuracy={r['accuracy']:.4f}  "
              f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")

    return best["algorithm"]


def tune_and_save(X_train, y_train):
    best_algorithm_name = select_best_model()

    if best_algorithm_name not in CANDIDATE_MODELS:
        raise ValueError(
            f"'{best_algorithm_name}' from {CV_RESULTS_PATH} not found in "
            f"CANDIDATE_MODELS — keep this dict in sync with 03_cross_validation.py."
        )

    base_pipeline = joblib.load(MODEL_PATH)
    preprocessing_steps = base_pipeline.steps[:-1]

    winning_classifier = clone(CANDIDATE_MODELS[best_algorithm_name])
    final_pipeline = build_pipeline(preprocessing_steps, winning_classifier)

    print(f"Refitting {best_algorithm_name} on full training data...")
    final_pipeline.fit(X_train, y_train)

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved winning pipeline ({best_algorithm_name}) to {MODEL_PATH}")

    return final_pipeline, best_algorithm_name


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']
    tune_and_save(X_train, y_train)
