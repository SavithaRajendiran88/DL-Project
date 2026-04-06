"""
model_mlp.py
============
Multi-Layer Perceptron (MLP) for wildlife movement anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from preprocessing import FEATURES, WINDOW


# ── Build ─────────────────────────────────────────────────────────────────────

def build_mlp(input_dim: int) -> models.Sequential:
    """
    Two-hidden-layer MLP with BatchNorm, Dropout, and L2 regularisation.

    Parameters
    ----------
    input_dim : flattened feature size (WINDOW × n_features)
    """
    tf.random.set_seed(42)

    mlp = models.Sequential([
        layers.Dense(
            128, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            input_shape=(input_dim,)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(
            64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(1, activation="sigmoid"),
    ])

    mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return mlp


# ── Reshape helper ────────────────────────────────────────────────────────────

def flatten_windows(*arrays: np.ndarray) -> tuple:
    """Flatten (N, WINDOW, F) → (N, WINDOW*F) for each array."""
    return tuple(a.reshape(a.shape[0], -1) for a in arrays)


# ── Train ─────────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Flatten inputs, build, and train the MLP. Returns (model, history)."""
    X_train_flat, X_val_flat = flatten_windows(X_train, X_val)

    mlp = build_mlp(input_dim=X_train_flat.shape[1])

    history = mlp.fit(
        X_train_flat, y_train,
        validation_data=(X_val_flat, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ],
    )
    return mlp, history, X_val_flat   # also return flat val for convenience


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("MLP Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("MLP Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_mlp(
    mlp: models.Sequential,
    X_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    X_test_flat    = flatten_windows(X_test)[0]
    y_pred_prob    = mlp.predict(X_test_flat)
    y_pred_binary  = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    mlp, history, _ = train_mlp(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_history(history)

    y_prob_mlp, y_pred_mlp = predict_mlp(mlp, splits["X_test"])
    print("MLP Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_mlp))
