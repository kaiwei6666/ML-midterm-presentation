import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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


RESULTS_OUTPUT_PATH = Path(__file__).resolve().parent / "ensemble_learning_results.csv"
POSITIVE_LABEL = "High Valence"


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


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_models() -> dict[str, object]:
    voting_estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier()),
    ]

    stacking_estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier()),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

    return {
        # Bagging methods are defined here.
        "BaggingClassifier": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        # Boosting methods are defined here.
        "AdaBoostClassifier": AdaBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        # Voting is executed here with soft voting.
        "VotingClassifier (Soft)": VotingClassifier(
            estimators=voting_estimators,
            voting="soft",
        ),
        # Stacking is executed here with Logistic Regression as the meta-model.
        "StackingClassifier": StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            n_jobs=-1,
        ),
    }


def get_auc(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    binary_y_test = (y_test == POSITIVE_LABEL).astype(int)
    model = pipeline.named_steps["model"]

    if hasattr(pipeline, "predict_proba"):
        positive_class_index = list(model.classes_).index(POSITIVE_LABEL)
        scores = pipeline.predict_proba(x_test)[:, positive_class_index]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_test)
        if getattr(model, "classes_", [POSITIVE_LABEL])[-1] != POSITIVE_LABEL:
            scores = -scores
    else:
        return np.nan

    return float(roc_auc_score(binary_y_test, scores))


def evaluate_model(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    pipeline = build_pipeline(model)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    auc_score = get_auc(pipeline, x_test, y_test)

    return {
        "Model": model_name,
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

    for model_name, model in build_models().items():
        results.append(
            evaluate_model(
                model_name=model_name,
                model=model,
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
    print("===== Ensemble Learning Results =====")
    print(results_frame.to_csv(index=False, lineterminator="\n").strip())
    print()

    best_result = results_frame.iloc[0]
    print("===== Best Ensemble Learning Result =====")
    print(f"Model: {best_result['Model']}")
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
