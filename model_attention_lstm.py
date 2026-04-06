"""
model_attention_lstm.py
=======================
Bidirectional LSTM with a custom attention mechanism for wildlife movement
anomaly detection. The attention layer weights each timestep's contribution
to the final anomaly prediction, providing interpretability over the 24-hour
window.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout,
    Multiply, Lambda, Activation, Input,
)
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from preprocessing import WINDOW


# ── Build ─────────────────────────────────────────────────────────────────────

def build_attention_lstm(
    seq_len: int = WINDOW,
    n_features: int = 4,
) -> Model:
    """
    Bidirectional LSTM + soft-attention classification model.

    Architecture
    ------------
    1. Bidirectional LSTM (returns all timestep outputs).
    2. A single Dense(1, tanh) layer scores each timestep → softmax weights.
    3. Weighted sum (context vector) fed into a Dense classification head.

    Parameters
    ----------
    seq_len    : window length (number of timesteps)
    n_features : number of features per timestep
    """
    inp = Input(shape=(seq_len, n_features), name="sequence_input")

    # Bidirectional LSTM — keep all timestep outputs for attention
    lstm_out = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name="bi_lstm",
    )(inp)

    # Attention scores & softmax weights
    attention_scores  = Dense(1, activation="tanh", name="attn_score")(lstm_out)
    attention_weights = Activation("softmax", name="attn_weights")(attention_scores)

    # Context vector: weighted sum over timesteps
    context = Multiply(name="context_vector")([lstm_out, attention_weights])
    context = Lambda(lambda x: K.sum(x, axis=1), name="context_sum")(context)

    # Classification head
    x   = Dense(64, activation="relu", name="fc1")(context)
    x   = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="AttentionLSTM")
    return model


# ── Train ─────────────────────────────────────────────────────────────────────

def train_attention_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Build, compile, and train the Attention-LSTM. Returns (model, history)."""
    attn_model = build_attention_lstm(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2],
    )
    attn_model.summary()

    attn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    history = attn_model.fit(
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
    return attn_model, history


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Attention-LSTM Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Attention-LSTM Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_attention_lstm(
    attn_model: Model,
    X_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    y_pred_prob   = attn_model.predict(X_test)
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    attn_model, history = train_attention_lstm(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        class_weights,
    )
    plot_history(history)

    y_prob_attn, y_pred_attn = predict_attention_lstm(
        attn_model, splits["X_test"]
    )
    print("Attention-LSTM Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_attn))
