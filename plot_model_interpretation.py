from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from music_valence_classification import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LABEL_ORDER,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    load_dataset,
)


LR_COEFFICIENT_OUTPUT_PATH = Path("logistic_regression_coefficients.png")
LR_ROC_OUTPUT_PATH = Path("logistic_regression_roc_curve.png")
SVM_ROC_OUTPUT_PATH = Path("svm_roc_curve.png")
SVM_PERFORMANCE_OUTPUT_PATH = Path("svm_performance_overview.png")


def train_pipeline(model: object, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def get_original_feature_importance(
    preprocessor: object,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    rows = []
    coefficient_index = 0

    for column in NUMERIC_COLUMNS:
        rows.append(
            {
                "Feature": column,
                "Coefficient Strength": abs(coefficients[coefficient_index]),
                "Aggregation": "single standardized coefficient",
            }
        )
        coefficient_index += 1

    for column in CATEGORICAL_COLUMNS:
        category_count = len(
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .categories_[CATEGORICAL_COLUMNS.index(column)]
        )
        grouped_coefficients = coefficients[coefficient_index : coefficient_index + category_count]
        rows.append(
            {
                "Feature": column,
                "Coefficient Strength": np.sqrt(np.mean(grouped_coefficients**2)),
                "Aggregation": "RMS of one-hot coefficients",
            }
        )
        coefficient_index += category_count

    return pd.DataFrame(rows).sort_values("Coefficient Strength", ascending=False)


def plot_logistic_regression_coefficients(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    coefficients = model.coef_[0]

    # In binary Logistic Regression, coef_ is for classes_[1].  The labels sort
    # as ["High Valence", "Low Valence"], so invert signs to make positive
    # coefficients indicate stronger association with High Valence.
    if model.classes_[1] != "High Valence":
        coefficients = -coefficients

    coefficient_frame = get_original_feature_importance(preprocessor, coefficients)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = sns.color_palette("Blues_r", n_colors=len(coefficient_frame))
    sns.barplot(
        data=coefficient_frame,
        x="Coefficient Strength",
        y="Feature",
        hue="Feature",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Logistic Regression: Original Feature Coefficient Strength",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_xlabel("Coefficient strength (absolute / grouped value)")
    ax.set_ylabel("")
    fig.text(
        0.5,
        0.02,
        f"Accuracy = {accuracy:.4f} | Macro F1 = {macro_f1:.4f}\n"
        "key and mode are grouped with the RMS of their one-hot encoded coefficients.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(LR_COEFFICIENT_OUTPUT_PATH, dpi=300, bbox_inches="tight")


def plot_logistic_regression_roc_curve(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    model = pipeline.named_steps["model"]

    positive_label = "High Valence"
    positive_class_index = list(model.classes_).index(positive_label)
    positive_scores = pipeline.predict_proba(x_test)[:, positive_class_index]
    binary_y_test = (y_test == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(binary_y_test, positive_scores)
    auc_score = roc_auc_score(binary_y_test, positive_scores)

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#1F5C99", linewidth=2.7, label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#8A8A8A", linestyle="--", linewidth=1.5, label="Random guess")
    ax.fill_between(fpr, tpr, alpha=0.18, color="#1F5C99")
    ax.set_title(
        "Logistic Regression: ROC Curve",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.96,
        f"Positive class: {positive_label}",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(LR_ROC_OUTPUT_PATH, dpi=300, bbox_inches="tight")


def plot_svm_performance_overview(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    predictions = pipeline.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    class_metrics = pd.DataFrame(
        {
            "Class": LABEL_ORDER,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }
    ).melt(id_vars="Class", var_name="Metric", value_name="Score")

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.25]})

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["High\nValence", "Low\nValence"],
        yticklabels=["High\nValence", "Low\nValence"],
        ax=axes[0],
    )
    axes[0].set_title("SVM Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0)

    sns.barplot(
        data=class_metrics,
        x="Class",
        y="Score",
        hue="Metric",
        palette=["#1F5C99", "#5B9FC9", "#A7CFE8"],
        ax=axes[1],
    )
    axes[1].set_title("SVM Class-wise Metrics", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    fig.suptitle(
        "SVM: Test-set Performance Overview",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        f"Accuracy = {accuracy:.4f} | Macro F1 = {macro_f1:.4f}. "
        "This view uses final test-set predictions instead of a 2D PCA projection.",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(SVM_PERFORMANCE_OUTPUT_PATH, dpi=300, bbox_inches="tight")


def plot_svm_roc_curve(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    model = pipeline.named_steps["model"]

    positive_label = "High Valence"
    decision_scores = pipeline.decision_function(x_test)
    if model.classes_[1] != positive_label:
        decision_scores = -decision_scores

    binary_y_test = (y_test == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(binary_y_test, decision_scores)
    auc_score = roc_auc_score(binary_y_test, decision_scores)

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#8A5A44", linewidth=2.7, label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#8A8A8A", linestyle="--", linewidth=1.5, label="Random guess")
    ax.fill_between(fpr, tpr, alpha=0.18, color="#8A5A44")
    ax.set_title(
        "SVM: ROC Curve",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.96,
        f"Positive class: {positive_label}\nScore source: SVM decision_function",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(SVM_ROC_OUTPUT_PATH, dpi=300, bbox_inches="tight")


def main() -> None:
    dataset, _, _ = load_dataset(DATASET_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[FEATURE_COLUMNS],
        dataset[LABEL_COLUMN],
        test_size=TEST_SIZE,
        stratify=dataset[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    lr_pipeline = train_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        x_train,
        y_train,
    )
    svm_pipeline = train_pipeline(
        SVC(kernel="rbf", random_state=RANDOM_STATE),
        x_train,
        y_train,
    )

    plot_logistic_regression_coefficients(lr_pipeline, x_test, y_test)
    plot_logistic_regression_roc_curve(lr_pipeline, x_test, y_test)
    plot_svm_performance_overview(svm_pipeline, x_test, y_test)
    plot_svm_roc_curve(svm_pipeline, x_test, y_test)

    print(f"Saved Logistic Regression coefficients plot to: {LR_COEFFICIENT_OUTPUT_PATH.resolve()}")
    print(f"Saved Logistic Regression ROC curve to: {LR_ROC_OUTPUT_PATH.resolve()}")
    print(f"Saved SVM performance overview to: {SVM_PERFORMANCE_OUTPUT_PATH.resolve()}")
    print(f"Saved SVM ROC curve to: {SVM_ROC_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
