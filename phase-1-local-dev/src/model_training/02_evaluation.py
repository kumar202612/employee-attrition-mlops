import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from config.paths import PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, MODEL_PATH, METRICS_PATH, ARTIFACT_DIR


def get_feature_names(pipeline, original_columns):
    """Extract feature names in the order ColumnTransformer outputs them."""
    ct = pipeline.named_steps['preprocessor']
    feature_names = []
    for name, transformer, indices in ct.transformers_:
        if name == 'remainder':
            feature_names.extend([original_columns[i] for i in indices])
        else:
            feature_names.extend([original_columns[i] for i in indices])
    return feature_names


def evaluate_data(X_train, y_train, X_test, y_test):
    # load pipeline
    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # train/test scores
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print('train score %: ', train_score * 100)
    print('test score %: ', test_score * 100)

    # feature importance — works for both linear (coef_) and tree-based (feature_importances_) models
    feature_names = get_feature_names(pipeline, X_train.columns.tolist())
    classifier = pipeline.named_steps['classifier']
    algorithm_name = type(classifier).__name__

    if hasattr(classifier, 'coef_'):
        importances = classifier.coef_[0]
        importance_label = "Coefficient"
    elif hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        importance_label = "Importance"
    else:
        importances = None
        importance_label = None

    if importances is not None:
        imp_df = pd.DataFrame({'Feature': feature_names, importance_label: importances})
        imp_df['Abs_' + importance_label] = np.abs(imp_df[importance_label])
        imp_df = imp_df.sort_values(by='Abs_' + importance_label, ascending=False)
        print(imp_df)

        plt.figure(figsize=(10, 6))
        plt.barh(imp_df['Feature'], imp_df[importance_label])
        plt.gca().invert_yaxis()
        plt.xlabel(importance_label)
        plt.title(f"{algorithm_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "feature_importance.png")
    else:
        print(f"No coef_ or feature_importances_ available for {algorithm_name}; skipping importance plot.")

    # save metrics for the final chosen model
    metrics = {
        "algorithm": algorithm_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "train_score": float(train_score),
        "test_score": float(test_score),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    return recall


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']
    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']
    evaluate_data(X_train, y_train, X_test, y_test)
