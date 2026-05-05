from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_PATH = Path("baseline_confusion_matrix_heatmaps.png")

CONFUSION_MATRICES = {
    "Decision Tree": [[66, 32], [33, 60]],
    "Logistic Regression": [[82, 16], [28, 65]],
    "Support Vector Machine (SVM)": [[83, 15], [29, 64]],
    "Perceptron": [[76, 22], [33, 60]],
    "K-Nearest Neighbors (KNN)": [[68, 30], [32, 61]],
}

CLASS_LABELS = ["High Valence", "Low Valence"]


def main() -> None:
    sns.set_theme(style="white", font_scale=1.05)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for ax, (model_name, matrix) in zip(axes, CONFUSION_MATRICES.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            linewidths=0.5,
            linecolor="white",
            square=True,
            ax=ax,
            annot_kws={"fontsize": 13, "fontweight": "bold"},
        )
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

    axes[-1].axis("off")
    fig.suptitle(
        "Baseline Model Confusion Matrix Heatmaps",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix heatmaps to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
