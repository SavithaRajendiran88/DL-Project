"""
evaluation.py
=============
Unified evaluation, comparison plots, and final summary for all models.

Expects a `model_registry` dict with the structure:
    {
        'ModelName': (y_pred_binary, y_pred_prob),
        ...
    }
where both arrays are 1-D and aligned with `y_test`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
)


# ── Metrics table ─────────────────────────────────────────────────────────────

def compute_metrics(
    model_registry: dict,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compute Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC for every
    model in the registry.

    Parameters
    ----------
    model_registry : dict  {name: (y_pred_binary, y_pred_prob)}
    y_test         : ground-truth labels

    Returns
    -------
    metrics_df : DataFrame sorted by F1-Score (descending)
    """
    rows = []
    for name, (y_pred, y_prob) in model_registry.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),                    4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0),  4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0),     4),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0),         4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),                     4),
            "PR-AUC":    round(average_precision_score(y_test, y_prob),           4),
        })

    metrics_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    print("=" * 72)
    print("                  MODEL COMPARISON — EVALUATION METRICS")
    print("=" * 72)
    print(metrics_df.to_string(index=False))
    print("=" * 72)
    return metrics_df


# ── Confusion matrices ────────────────────────────────────────────────────────

def plot_confusion_matrices(
    model_registry: dict,
    y_test: np.ndarray,
) -> None:
    """Plot one confusion matrix per model in a 2×4 grid."""
    n       = len(model_registry)
    n_cols  = 4
    n_rows  = (n + n_cols - 1) // n_cols   # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")

    ax_flat = axes.flatten()

    for ax, (name, (y_pred, _)) in zip(ax_flat, model_registry.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Normal", "Anomaly"],
            colorbar=False,
            ax=ax,
            cmap="Blues",
        )
        ax.set_title(name, fontsize=11)

    # Hide any unused subplots
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    model_registry: dict,
    y_test: np.ndarray,
) -> None:
    """Overlay ROC curves for all models."""
    plt.figure(figsize=(9, 7))

    for name, (_, y_prob) in model_registry.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name}  (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Precision-Recall curves ───────────────────────────────────────────────────

def plot_pr_curves(
    model_registry: dict,
    y_test: np.ndarray,
) -> None:
    """Overlay Precision-Recall curves for all models."""
    plt.figure(figsize=(9, 7))

    for name, (_, y_prob) in model_registry.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap           = average_precision_score(y_test, y_prob)
        plt.plot(rec, prec, label=f"{name}  (AP={ap:.3f})")

    baseline = y_test.mean()
    plt.axhline(
        baseline, color="k", linestyle="--", linewidth=1,
        label=f"Random Baseline (precision={baseline:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── F1 vs ROC-AUC bar chart ───────────────────────────────────────────────────

def plot_f1_vs_auc(metrics_df: pd.DataFrame) -> None:
    """Side-by-side bar chart of F1-Score and ROC-AUC for each model."""
    fig, ax = plt.subplots(figsize=(11, 5))

    x     = np.arange(len(metrics_df))
    width = 0.35

    ax.bar(x - width / 2, metrics_df["F1-Score"], width,
           label="F1-Score", color="steelblue")
    ax.bar(x + width / 2, metrics_df["ROC-AUC"],  width,
           label="ROC-AUC",  color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — F1-Score vs ROC-AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, row in metrics_df.reset_index(drop=True).iterrows():
        ax.text(i - width / 2, row["F1-Score"] + 0.01,
                f"{row['F1-Score']:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["ROC-AUC"] + 0.01,
                f"{row['ROC-AUC']:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


# ── Final narrative summary ───────────────────────────────────────────────────

def print_summary(metrics_df: pd.DataFrame) -> None:
    """Print a human-readable final summary of results."""
    best = metrics_df.iloc[0]
    print("\n── Wildlife Movement Anomaly Detection — Final Summary ──────────")
    print("Dataset      : Movebank Etosha Elephant GPS (4 individuals, 2010)")
    print("Window size  : 24 timesteps  |  Anomaly label: top 5% speed")
    print("Train elephants: LA11, LA12  |  Val: LA13  |  Test: LA14")
    print()
    print(f"Best model by F1-Score: {best['Model']}")
    print(f"  Accuracy : {best['Accuracy']}")
    print(f"  Precision: {best['Precision']}")
    print(f"  Recall   : {best['Recall']}")
    print(f"  F1-Score : {best['F1-Score']}")
    print(f"  ROC-AUC  : {best['ROC-AUC']}")
    print(f"  PR-AUC   : {best['PR-AUC']}")
    print()
    print("All models trained with class-weight balancing and early stopping.")
    print("Attention-LSTM adds interpretability via per-timestep attention weights.")
    print("LSTM+Embedding conditions predictions on individual elephant identity.")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(
    model_registry: dict,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Run the full evaluation suite:
      1. Metrics table
      2. Confusion matrices
      3. ROC curves
      4. Precision-Recall curves
      5. F1 vs AUC bar chart
      6. Narrative summary

    Returns metrics_df for downstream use.
    """
    metrics_df = compute_metrics(model_registry, y_test)
    plot_confusion_matrices(model_registry, y_test)
    plot_roc_curves(model_registry, y_test)
    plot_pr_curves(model_registry, y_test)
    plot_f1_vs_auc(metrics_df)
    print_summary(metrics_df)
    return metrics_df


# ── Entrypoint (demo with dummy data) ────────────────────────────────────────

if __name__ == "__main__":
    # To run a real evaluation, import predictions from each model module and
    # assemble the registry as shown below, then call run_evaluation().
    print("Import this module and call run_evaluation(model_registry, y_test).")
    print("Example:")
    print("  from evaluation import run_evaluation")
    print("  model_registry = {")
    print("      'MLP':  (y_pred_mlp.ravel(),  y_prob_mlp.ravel()),")
    print("      'CNN':  (y_pred_cnn.ravel(),  y_prob_cnn.ravel()),")
    print("      ...}")
    print("  metrics_df = run_evaluation(model_registry, y_test)")
