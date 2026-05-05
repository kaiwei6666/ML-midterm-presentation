import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from music_valence_classification import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    load_dataset,
)


RESULTS_OUTPUT_PATH = Path(__file__).resolve().parent / "hyperparameter_tuning_results.csv"
POSITIVE_LABEL = "High Valence"
CV_FOLDS = 5
SCORING = "f1_macro"
N_JOBS = -1


def build_shared_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset, _, _ = load_dataset(DATASET_PATH)
    features = dataset[FEATURE_COLUMNS]
    labels = dataset[LABEL_COLUMN]
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )


def build_search_configs() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
                "model__max_iter": [1000],
            },
        ),
        "SVM": (
            SVC(random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
                "model__probability": [True],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 3, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "Perceptron": (
            Perceptron(random_state=RANDOM_STATE),
            {
                "model__penalty": [None, "l2", "l1", "elasticnet"],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__max_iter": [1000, 2000],
                "model__eta0": [0.1, 1.0],
            },
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def get_auc(best_pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    binary_y_test = (y_test == POSITIVE_LABEL).astype(int)
    model = best_pipeline.named_steps["model"]

    if hasattr(best_pipeline, "predict_proba"):
        positive_class_index = list(model.classes_).index(POSITIVE_LABEL)
        scores = best_pipeline.predict_proba(x_test)[:, positive_class_index]
    elif hasattr(best_pipeline, "decision_function"):
        scores = best_pipeline.decision_function(x_test)
        if getattr(model, "classes_", [POSITIVE_LABEL])[-1] != POSITIVE_LABEL:
            scores = -scores
    else:
        return np.nan

    return float(roc_auc_score(binary_y_test, scores))


def evaluate_model(
    model_name: str,
    model: object,
    param_grid: dict[str, list[object]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    pipeline = build_pipeline(model)

    # Each model keeps the same baseline preprocessing pipeline and only tunes
    # the estimator-specific hyperparameters listed in param_grid.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=N_JOBS,
    )
    grid_search.fit(x_train, y_train)

    best_pipeline = grid_search.best_estimator_
    predictions = best_pipeline.predict(x_test)
    auc_score = get_auc(best_pipeline, x_test, y_test)

    best_params = {
        key.replace("model__", ""): value for key, value in grid_search.best_params_.items()
    }

    return {
        "Model": model_name,
        "Best Parameters": json.dumps(best_params, ensure_ascii=True, sort_keys=True),
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, average="macro", zero_division=0),
            4,
        ),
        "Recall": round(
            recall_score(y_test, predictions, average="macro", zero_division=0),
            4,
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, average="macro", zero_division=0),
            4,
        ),
        "AUC": round(auc_score, 4) if not np.isnan(auc_score) else np.nan,
    }


def run_evaluation() -> pd.DataFrame:
    x_train, x_test, y_train, y_test = build_shared_split()
    results = []

    for model_name, (model, param_grid) in build_search_configs().items():
        results.append(
            evaluate_model(
                model_name=model_name,
                model=model,
                param_grid=param_grid,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

    results_frame = pd.DataFrame(results)
    results_frame = results_frame.sort_values(
        by=["F1 Score", "Accuracy", "AUC"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return results_frame


def print_results(results_frame: pd.DataFrame) -> None:
    print("===== Hyperparameter Tuning Results =====")
    print(results_frame.to_csv(index=False, lineterminator="\n").strip())
    print()

    best_result = results_frame.iloc[0]
    print("===== Best Hyperparameter Tuning Result =====")
    print(f"Model: {best_result['Model']}")
    print(f"Best Parameters: {best_result['Best Parameters']}")
    print(f"Accuracy: {best_result['Accuracy']}")
    print(f"Precision: {best_result['Precision']}")
    print(f"Recall: {best_result['Recall']}")
    print(f"F1 Score: {best_result['F1 Score']}")
    print(f"AUC: {best_result['AUC']}")


def main() -> None:
    results_frame = run_evaluation()
    results_frame.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print_results(results_frame)


if __name__ == "__main__":
    main()
