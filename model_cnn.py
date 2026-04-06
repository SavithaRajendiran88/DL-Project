"""
model_cnn.py
============
1-D Convolutional Neural Network (CNN) for wildlife movement anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from preprocessing import WINDOW


# ── Build ─────────────────────────────────────────────────────────────────────

def build_cnn(window: int = WINDOW, n_features: int = 4) -> models.Sequential:
    """
    Three-block Conv1D network with GlobalAveragePooling and a Dense head.

    Parameters
    ----------
    window     : number of timesteps per window
    n_features : number of input features per timestep
    """
    cnn = models.Sequential([
        layers.Conv1D(
            32, kernel_size=5, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            input_shape=(window, n_features)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(),

        layers.Conv1D(
            64, kernel_size=3, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(),

        layers.Conv1D(
            128, kernel_size=3, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return cnn


# ── Train ─────────────────────────────────────────────────────────────────────

def train_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Build and train the CNN. Returns (model, history)."""
    cnn = build_cnn(window=X_train.shape[1], n_features=X_train.shape[2])

    history = cnn.fit(
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
    return cnn, history


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_cnn(
    cnn: models.Sequential,
    X_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    y_pred_prob   = cnn.predict(X_test)
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    cnn, history = train_cnn(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_history(history)

    y_prob_cnn, y_pred_cnn = predict_cnn(cnn, splits["X_test"])
    print("CNN Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_cnn))
