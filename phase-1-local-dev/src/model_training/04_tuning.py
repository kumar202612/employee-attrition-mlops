import json
import pandas as pd
import joblib
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config.paths import PREPROCESSED_TRAIN_PATH, MODEL_PATH, ARTIFACT_DIR

CV_RESULTS_PATH = ARTIFACT_DIR / "cv_results.json"
TUNING_RESULTS_PATH = ARTIFACT_DIR / "tuning_results.json"

# Must match the CANDIDATE_MODELS keys/instances used in 03_cross_validation.py
CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

# Hyperparameter search space per algorithm. Only the winning algorithm's
# grid is actually searched — keep these in sync with CANDIDATE_MODELS keys.
PARAM_DISTRIBUTIONS = {
    "LogisticRegression": {
        "classifier__C": uniform(0.01, 10),
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "RandomForest": {
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(3, 25),
        "classifier__min_samples_split": randint(2, 10),
        "classifier__min_samples_leaf": randint(1, 8),
        "classifier__max_features": ["sqrt", "log2", None],
    },
    "GradientBoosting": {
        "classifier__n_estimators": randint(100, 400),
        "classifier__learning_rate": uniform(0.01, 0.29),
        "classifier__max_depth": randint(2, 8),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__min_samples_leaf": randint(1, 8),
    },
}

# Which metric decides both the algorithm winner (from CV results) and the
# search objective here. Recall is prioritized by default since missing an
# at-risk employee (false negative) is usually costlier than flagging a
# safe one (false positive) in attrition prediction.
SELECTION_METRIC = "recall"

N_ITER = 30       # number of random hyperparameter combinations to try
CV_FOLDS = 5
RANDOM_STATE = 42


def build_pipeline(preprocessing_steps, classifier):
    return Pipeline(
        steps=[(name, clone(step)) for name, step in preprocessing_steps]
        + [("classifier", classifier)]
    )


def select_best_algorithm():
    if not CV_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{CV_RESULTS_PATH} not found — run 03_cross_validation.py first."
        )
    with open(CV_RESULTS_PATH) as f:
        cv_results = json.load(f)

    best = max(cv_results, key=lambda r: r[SELECTION_METRIC])
    print(f"Best algorithm by {SELECTION_METRIC} (pre-tuning CV): {best['algorithm']} "
          f"({SELECTION_METRIC}={best[SELECTION_METRIC]:.4f})")
    print("Full comparison:")
    for r in sorted(cv_results, key=lambda r: r[SELECTION_METRIC], reverse=True):
        print(f"  {r['algorithm']:<20} accuracy={r['accuracy']:.4f}  "
              f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")
    return best["algorithm"]


def tune_and_save(X_train, y_train):
    best_algorithm_name = select_best_algorithm()

    if best_algorithm_name not in CANDIDATE_MODELS:
        raise ValueError(
            f"'{best_algorithm_name}' from {CV_RESULTS_PATH} not found in "
            f"CANDIDATE_MODELS — keep this dict in sync with 03_cross_validation.py."
        )
    if best_algorithm_name not in PARAM_DISTRIBUTIONS:
        raise ValueError(
            f"No PARAM_DISTRIBUTIONS entry for '{best_algorithm_name}' — "
            f"add a hyperparameter search space for this algorithm."
        )

    base_pipeline = joblib.load(MODEL_PATH)
    preprocessing_steps = base_pipeline.steps[:-1]
    candidate_classifier = clone(CANDIDATE_MODELS[best_algorithm_name])
    search_pipeline = build_pipeline(preprocessing_steps, candidate_classifier)

    print(f"\nRunning RandomizedSearchCV on {best_algorithm_name} "
          f"({N_ITER} candidates, {CV_FOLDS}-fold CV, scoring={SELECTION_METRIC})...")

    search = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[best_algorithm_name],
        n_iter=N_ITER,
        scoring=SELECTION_METRIC,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,  # automatically refits the best combo on the full X_train, y_train
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest {SELECTION_METRIC} score from search: {search.best_score_:.4f}")
    print("Best hyperparameters found:")
    for param, value in search.best_params_.items():
        print(f"  {param} = {value}")

    final_pipeline = search.best_estimator_
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved tuned pipeline ({best_algorithm_name}) to {MODEL_PATH}")

    # Persist tuning results for traceability / demo purposes
    tuning_summary = {
        "algorithm": best_algorithm_name,
        "selection_metric": SELECTION_METRIC,
        "best_cv_score": search.best_score_,
        "best_params": search.best_params_,
        "n_iter": N_ITER,
        "cv_folds": CV_FOLDS,
    }
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print(f"Saved tuning summary to {TUNING_RESULTS_PATH}")

    return final_pipeline, best_algorithm_name


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']
    tune_and_save(X_train, y_train)
