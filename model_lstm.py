"""
model_lstm.py
=============
Long Short-Term Memory (LSTM) for wildlife movement anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense, Dropout

from preprocessing import WINDOW


# ── Build ─────────────────────────────────────────────────────────────────────

def build_lstm(window: int = WINDOW, n_features: int = 4) -> models.Sequential:
    """
    Single LSTM layer followed by a Dense classification head.

    Parameters
    ----------
    window     : number of timesteps per window
    n_features : number of input features per timestep
    """
    lstm_model = models.Sequential([
        LSTM(
            64,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            input_shape=(window, n_features)
        ),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return lstm_model


# ── Train ─────────────────────────────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Build and train the LSTM. Returns (model, history)."""
    lstm_model = build_lstm(window=X_train.shape[1], n_features=X_train.shape[2])

    history = lstm_model.fit(
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
    return lstm_model, history


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("LSTM Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("LSTM Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_lstm(
    lstm_model: models.Sequential,
    X_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    y_pred_prob   = lstm_model.predict(X_test)
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    lstm_model, history = train_lstm(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_history(history)

    y_prob_lstm, y_pred_lstm = predict_lstm(lstm_model, splits["X_test"])
    print("LSTM Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_lstm))
