import json
import pandas as pd
import joblib
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config.paths import PREPROCESSED_TRAIN_PATH, MODEL_PATH, ARTIFACT_DIR

CV_RESULTS_PATH = ARTIFACT_DIR / "cv_results.json"

CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}


def build_pipeline(preprocessing_steps, classifier):
    """Re-use the already-fitted preprocessing steps, swap in a new classifier."""
    return Pipeline(
        steps=[(name, clone(step)) for name, step in preprocessing_steps]
        + [("classifier", classifier)]
    )


def cv_data(X_train, y_train):
    base_pipeline = joblib.load(MODEL_PATH)
    preprocessing_steps = base_pipeline.steps[:-1]  # everything except the final classifier
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, clf in CANDIDATE_MODELS.items():
        pipeline = build_pipeline(preprocessing_steps, clf)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=strat_cv,
            scoring=["accuracy", "precision", "recall", "f1"],
        )
        result = {
            "algorithm": name,
            "accuracy": float(scores["test_accuracy"].mean()),
            "precision": float(scores["test_precision"].mean()),
            "recall": float(scores["test_recall"].mean()),
            "f1": float(scores["test_f1"].mean()),
        }
        results.append(result)
        print(f"{name}: accuracy={result['accuracy']*100:.2f}%  recall={result['recall']*100:.2f}%")

    with open(CV_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved CV comparison to {CV_RESULTS_PATH}")

    return results


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']
    cv_data(X_train, y_train)
