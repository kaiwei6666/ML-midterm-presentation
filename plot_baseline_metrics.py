from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS_PATH = Path("valence_classification_metrics.csv")
OUTPUT_PATH = Path("baseline_model_metrics_combined.png")


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    metric_columns = [
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-score (Macro)",
    ]
    long_metrics = metrics.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    palette = {
        "Decision Tree": "#8FB3A1",
        "Logistic Regression": "#D98C5F",
        "Support Vector Machine (SVM)": "#6E95C7",
        "Perceptron": "#C26A7A",
        "K-Nearest Neighbors (KNN)": "#B89D5E",
    }

    for ax, metric in zip(axes, metric_columns):
        subset = long_metrics[long_metrics["Metric"] == metric].sort_values(
            by="Score",
            ascending=False,
        )
        sns.barplot(
            data=subset,
            x="Score",
            y="Model",
            hue="Model",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(metric, fontsize=15, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.set_xlim(0.55, 0.80)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=3, fontsize=10)

    fig.suptitle(
        "Baseline Model Evaluation Metrics",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Task: High Valence vs Low Valence classification using selected music features",
        ha="center",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved combined metrics chart to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
