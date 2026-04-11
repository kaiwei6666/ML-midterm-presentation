import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = Path("spotify-2023.csv")
FEATURE_COLUMNS = [
    "bpm",
    "key",
    "mode",
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
TAIL_COLUMNS = [
    "bpm",
    "key",
    "mode",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
NUMERIC_COLUMNS = [
    "bpm",
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
CATEGORICAL_COLUMNS = ["key", "mode"]
TARGET_COLUMN = "valence_%"
LABEL_COLUMN = "valence_label"
LABEL_ORDER = ["High Valence", "Low Valence"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
METRICS_OUTPUT_PATH = Path("valence_classification_metrics.csv")
REPORT_OUTPUT_PATH = Path("valence_classification_report.md")


def format_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([str(row[column]) for column in frame.columns])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_dataset(path: Path) -> tuple[pd.DataFrame, float, pd.Series]:
    raw_lines = path.read_text(encoding="latin1").splitlines()
    parsed_rows = []
    for line in raw_lines[1:]:
        parts = line.split(",")
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) < len(TAIL_COLUMNS):
            continue
        parsed_rows.append(dict(zip(TAIL_COLUMNS, parts[-len(TAIL_COLUMNS) :])))

    dataset = pd.DataFrame(parsed_rows)

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    valence_median = dataset[TARGET_COLUMN].median()
    dataset[LABEL_COLUMN] = np.where(
        dataset[TARGET_COLUMN] >= valence_median,
        LABEL_ORDER[0],
        LABEL_ORDER[1],
    )

    missing_summary = dataset[FEATURE_COLUMNS].isna().sum()
    return dataset, valence_median, missing_summary


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def evaluate_models(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, int]]:
    features = dataset[FEATURE_COLUMNS]
    labels = dataset[LABEL_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    split_summary = {
        "train_size": len(x_train),
        "test_size": len(x_test),
    }
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Support Vector Machine (SVM)": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Perceptron": Perceptron(max_iter=2000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    }

    preprocessor = build_preprocessor()
    metrics_rows = []
    confusion_matrices: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)
        confusion_matrices[model_name] = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in LABEL_ORDER],
            columns=[f"Predicted {label}" for label in LABEL_ORDER],
        )
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "Precision (Macro)": round(
                    precision_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "Recall (Macro)": round(
                    recall_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "F1-score (Macro)": round(
                    f1_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(
        by=["F1-score (Macro)", "Accuracy"],
        ascending=False,
    )
    return metrics_frame, confusion_matrices, split_summary


def build_report(
    dataset: pd.DataFrame,
    valence_median: float,
    missing_summary: pd.Series,
    metrics_frame: pd.DataFrame,
    confusion_matrices: dict[str, pd.DataFrame],
    split_summary: dict[str, int],
) -> str:
    report_lines = [
        "# Music Valence Classification Report",
        "",
        "## Experiment Setup",
        f"- Dataset: `{DATASET_PATH.name}`",
        f"- Samples: {len(dataset)}",
        "- Parsing note: the source CSV has malformed quotes in some text columns, so the requested music features were extracted from the tail of each raw line to preserve valid records.",
        f"- Features: {', '.join(FEATURE_COLUMNS)}",
        f"- Label rule: `{TARGET_COLUMN} >= {valence_median:.1f}` -> `High Valence`, otherwise `Low Valence`",
        f"- Train/Test split: {split_summary['train_size']} / {split_summary['test_size']} (stratified, random_state=42)",
        "",
        "## Class Distribution",
    ]

    class_distribution = (
        dataset[LABEL_COLUMN]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    report_lines.extend(
        [
            format_markdown_table(class_distribution),
            "",
            "## Missing Values in Selected Features",
        ]
    )

    missing_frame = missing_summary.rename_axis("Feature").reset_index(name="Missing Count")
    report_lines.extend(
        [
            format_markdown_table(missing_frame),
            "",
            "## Model Comparison",
            format_markdown_table(metrics_frame),
            "",
            "## Confusion Matrices",
        ]
    )

    for model_name, matrix in confusion_matrices.items():
        report_lines.extend(
            [
                f"### {model_name}",
                format_markdown_table(matrix.reset_index().rename(columns={"index": " " })),
                "",
            ]
        )

    best_row = metrics_frame.iloc[0]
    report_lines.extend(
        [
            "## Conclusion",
            (
                f"- Best model: `{best_row['Model']}` with Accuracy `{best_row['Accuracy']}` "
                f"and Macro F1 `{best_row['F1-score (Macro)']}`."
            ),
            "- These audio features show moderate predictive power for valence, but they are not strong enough for highly reliable emotion classification on their own.",
        ]
    )

    return "\n".join(report_lines) + "\n"


def main() -> None:
    dataset, valence_median, missing_summary = load_dataset(DATASET_PATH)
    metrics_frame, confusion_matrices, split_summary = evaluate_models(dataset)

    metrics_frame.to_csv(METRICS_OUTPUT_PATH, index=False)
    report = build_report(
        dataset=dataset,
        valence_median=valence_median,
        missing_summary=missing_summary,
        metrics_frame=metrics_frame,
        confusion_matrices=confusion_matrices,
        split_summary=split_summary,
    )
    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"Median valence: {valence_median:.1f}")
    print(f"Saved metrics to: {METRICS_OUTPUT_PATH.resolve()}")
    print(f"Saved report to: {REPORT_OUTPUT_PATH.resolve()}")
    print()
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
