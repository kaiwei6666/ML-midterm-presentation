import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


RESULTS_OUTPUT_PATH = Path(__file__).resolve().parent / "pca_lda_results.csv"
POSITIVE_LABEL = "High Valence"


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Perceptron": Perceptron(max_iter=2000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }


def to_dense(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


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


def preprocess_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    return to_dense(x_train_processed), to_dense(x_test_processed)


def get_pca_component_range(x_train_processed: np.ndarray) -> range:
    max_components = min(x_train_processed.shape[0], x_train_processed.shape[1])
    return range(1, max_components + 1)


def get_auc(model: object, x_test: np.ndarray, y_test: pd.Series) -> float:
    binary_y_test = (y_test == POSITIVE_LABEL).astype(int)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
    else:
        return np.nan

    return float(roc_auc_score(binary_y_test, scores))


def evaluate_method(
    method_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict[str, object]]:
    rows = []

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        auc_score = get_auc(model, x_test, y_test)
        rows.append(
            {
                "Method": f"{method_name} + {model_name}",
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
        )

    return rows


def run_experiments() -> pd.DataFrame:
    x_train, x_test, y_train, y_test = build_shared_split()
    x_train_processed, x_test_processed = preprocess_features(x_train, x_test)

    results = []

    # PCA is applied after the baseline preprocessing and before model training.
    for n_components in get_pca_component_range(x_train_processed):
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        x_train_pca = pca.fit_transform(x_train_processed)
        x_test_pca = pca.transform(x_test_processed)
        results.extend(
            evaluate_method(
                method_name=f"PCA ({n_components})",
                x_train=x_train_pca,
                x_test=x_test_pca,
                y_train=y_train,
                y_test=y_test,
            )
        )

    # LDA is applied after the baseline preprocessing and before model training.
    lda = LinearDiscriminantAnalysis(n_components=1)
    x_train_lda = lda.fit_transform(x_train_processed, y_train)
    x_test_lda = lda.transform(x_test_processed)
    results.extend(
        evaluate_method(
            method_name="LDA (1)",
            x_train=x_train_lda,
            x_test=x_test_lda,
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
    print("===== PCA / LDA Results =====")
    print(results_frame.to_csv(index=False, lineterminator="\n").strip())
    print()

    best_result = results_frame.iloc[0]
    print("===== Best PCA / LDA Result =====")
    print(f"Method: {best_result['Method']}")
    print(f"Accuracy: {best_result['Accuracy']}")
    print(f"Precision: {best_result['Precision']}")
    print(f"Recall: {best_result['Recall']}")
    print(f"F1 Score: {best_result['F1 Score']}")
    print(f"AUC: {best_result['AUC']}")


def main() -> None:
    results_frame = run_experiments()
    results_frame.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print_results(results_frame)


if __name__ == "__main__":
    main()
