"""
model_gru.py
============
Gated Recurrent Unit (GRU) for wildlife movement anomaly detection,
including a grid-search hyperparameter experimentation routine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GRU, Dense, Dropout
from itertools import product

from preprocessing import WINDOW


# ── Build ─────────────────────────────────────────────────────────────────────

def build_gru(
    window: int = WINDOW,
    n_features: int = 4,
    learning_rate: float = 1e-3,
) -> models.Sequential:
    """
    Single GRU layer followed by a Dense classification head.

    Parameters
    ----------
    window        : number of timesteps per window
    n_features    : number of input features per timestep
    learning_rate : Adam learning rate
    """
    gru_model = models.Sequential([
        GRU(
            64,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            input_shape=(window, n_features)
        ),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return gru_model


# ── Train ─────────────────────────────────────────────────────────────────────

def train_gru(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Build and train the GRU. Returns (model, history)."""
    gru_model = build_gru(
        window=X_train.shape[1],
        n_features=X_train.shape[2],
        learning_rate=learning_rate,
    )

    history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ],
    )
    return gru_model, history


# ── Hyperparameter grid search ────────────────────────────────────────────────

HP_GRID = {
    "learning_rate": [1e-3, 1e-4],
    "batch_size":    [32, 64],
}


def hyperparameter_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    hp_grid: dict = HP_GRID,
    epochs: int = 20,
) -> pd.DataFrame:
    """
    Grid-search over learning rate × batch size for the GRU model.

    Returns a DataFrame of results sorted by val_auc (descending).
    """
    results = []

    for lr, bs in product(hp_grid["learning_rate"], hp_grid["batch_size"]):
        print(f"\n── GRU | lr={lr}  batch_size={bs} ──")

        gru_hp = build_gru(
            window=X_train.shape[1],
            n_features=X_train.shape[2],
            learning_rate=lr,
        )

        hist = gru_hp.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=bs,
            class_weight=class_weights,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=4, restore_best_weights=True
                )
            ],
        )

        best_epoch = int(np.argmin(hist.history["val_loss"]))
        val_acc    = hist.history["val_accuracy"][best_epoch]
        val_auc    = hist.history["val_AUC"][best_epoch]

        results.append({
            "learning_rate": lr,
            "batch_size":    bs,
            "best_epoch":    best_epoch + 1,
            "val_accuracy":  round(val_acc, 4),
            "val_auc":       round(val_auc, 4),
        })
        print(f"   Best epoch {best_epoch+1} → "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

    hp_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    print("\n── Hyperparameter Search Results ──")
    print(hp_df.to_string(index=False))
    return hp_df


def plot_hp_results(hp_df: pd.DataFrame) -> None:
    """Bar chart comparing val_accuracy and val_AUC across HP combinations."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [
        f"lr={r['learning_rate']}\nbs={r['batch_size']}"
        for _, r in hp_df.iterrows()
    ]
    x, width = np.arange(len(labels)), 0.35

    ax.bar(x - width / 2, hp_df["val_accuracy"], width, label="Val Accuracy")
    ax.bar(x + width / 2, hp_df["val_auc"],      width, label="Val AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("GRU Hyperparameter Experimentation (LR × Batch Size)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Plot training history ─────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("GRU Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("GRU Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_gru(
    gru_model: models.Sequential,
    X_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    y_pred_prob   = gru_model.predict(X_test)
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    # Default training run
    gru_model, history = train_gru(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_history(history)

    y_prob_gru, y_pred_gru = predict_gru(gru_model, splits["X_test"])
    print("GRU Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_gru))

    # Hyperparameter search
    hp_df = hyperparameter_search(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_hp_results(hp_df)
