"""
model_embedding_lstm.py
=======================
LSTM conditioned on a learnable elephant-identity embedding.

Each elephant (LA11, LA12, LA13, LA14) is assigned a trainable 4-dimensional
vector. This embedding is concatenated with the movement features at every
timestep, allowing the model to learn individual-specific movement patterns
alongside the shared temporal dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Concatenate, Reshape, Input,
    LSTM, Dense, Dropout, RepeatVector,
)
from tensorflow.keras import Model

from preprocessing import WINDOW


# ── Elephant ID encoding ──────────────────────────────────────────────────────

def build_elephant_vocab(df_model) -> dict:
    """
    Build a mapping from elephant string IDs to integer indices.

    Parameters
    ----------
    df_model : DataFrame with column 'individual-local-identifier'

    Returns
    -------
    elephant_to_idx : dict  {elephant_id: int}
    """
    unique_ids     = sorted(df_model["individual-local-identifier"].unique())
    elephant_to_idx = {e: i for i, e in enumerate(unique_ids)}
    print("Elephant vocabulary:", elephant_to_idx)
    return elephant_to_idx


def make_id_array(
    elephant_ids_arr: np.ndarray,
    elephant_to_idx: dict,
) -> np.ndarray:
    """Map an array of elephant string IDs → integer index array."""
    return np.array(
        [elephant_to_idx[e] for e in elephant_ids_arr], dtype=np.int32
    )


def make_split_id_arrays(
    window_elephants: np.ndarray,
    elephant_to_idx: dict,
    train_mask: np.ndarray,
    val_mask:   np.ndarray,
    test_mask:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eid_train, eid_val, eid_test) integer ID arrays."""
    eid_train = make_id_array(window_elephants[train_mask], elephant_to_idx)
    eid_val   = make_id_array(window_elephants[val_mask],   elephant_to_idx)
    eid_test  = make_id_array(window_elephants[test_mask],  elephant_to_idx)
    print("Embedding ID shape (train):", eid_train.shape)
    return eid_train, eid_val, eid_test


# ── Build ─────────────────────────────────────────────────────────────────────

def build_embedding_lstm(
    n_elephants: int,
    embed_dim: int = 4,
    window: int = WINDOW,
    n_features: int = 4,
) -> Model:
    """
    Dual-input model: movement sequence + elephant integer ID.

    The elephant embedding is expanded to (WINDOW, embed_dim) and
    concatenated with the (WINDOW, n_features) movement tensor before
    passing through an LSTM.

    Parameters
    ----------
    n_elephants : total number of unique elephant IDs
    embed_dim   : dimensionality of the elephant embedding
    window      : number of timesteps per window
    n_features  : movement features per timestep
    """
    seq_input = Input(shape=(window, n_features), name="movement_input")
    id_input  = Input(shape=(1,),                 name="elephant_id_input")

    # Learnable elephant embedding
    emb          = Embedding(
        input_dim=n_elephants, output_dim=embed_dim,
        name="elephant_embedding"
    )(id_input)
    emb          = Reshape((embed_dim,))(emb)
    emb_repeated = RepeatVector(window)(emb)   # (batch, WINDOW, embed_dim)

    # Concatenate with sequence features → (batch, WINDOW, n_features+embed_dim)
    merged = Concatenate(axis=-1)([seq_input, emb_repeated])

    # LSTM + head
    x   = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(merged)
    x   = Dense(32, activation="relu")(x)
    x   = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[seq_input, id_input], outputs=out,
                  name="LSTM_Embedding")
    return model


# ── Train ─────────────────────────────────────────────────────────────────────

def train_embedding_lstm(
    X_train: np.ndarray, eid_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, eid_val:   np.ndarray, y_val:   np.ndarray,
    n_elephants: int,
    class_weights: dict,
    embed_dim: int = 4,
    epochs: int = 30,
    batch_size: int = 64,
):
    """Build, compile, and train the Embedding-LSTM. Returns (model, history)."""
    embed_model = build_embedding_lstm(
        n_elephants=n_elephants,
        embed_dim=embed_dim,
        window=X_train.shape[1],
        n_features=X_train.shape[2],
    )
    embed_model.summary()

    embed_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    history = embed_model.fit(
        [X_train, eid_train], y_train,
        validation_data=([X_val, eid_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ],
    )
    return embed_model, history


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history) -> None:
    """Plot training/validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("LSTM + Embedding Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("LSTM + Embedding Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_embedding_lstm(
    embed_model: Model,
    X_test: np.ndarray,
    eid_test: np.ndarray,
    threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_prob, y_pred_binary) for the test set."""
    y_pred_prob   = embed_model.predict([X_test, eid_test])
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    return y_pred_prob, y_pred_binary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline
    from sklearn.metrics import confusion_matrix

    data = load_data()
    df_model, splits, window_elephants, class_weights, _ = run_pipeline(data)

    # Build vocabulary and ID arrays
    elephant_to_idx = build_elephant_vocab(df_model)
    n_elephants     = len(elephant_to_idx)

    eid_train, eid_val, eid_test = make_split_id_arrays(
        window_elephants,
        elephant_to_idx,
        splits["train_mask"],
        splits["val_mask"],
        splits["test_mask"],
    )

    embed_model, history = train_embedding_lstm(
        splits["X_train"], eid_train, splits["y_train"],
        splits["X_val"],   eid_val,   splits["y_val"],
        n_elephants=n_elephants,
        class_weights=class_weights,
    )
    plot_history(history)

    y_prob_embed, y_pred_embed = predict_embedding_lstm(
        embed_model, splits["X_test"], eid_test
    )
    print("LSTM + Embedding Confusion Matrix:\n",
          confusion_matrix(splits["y_test"], y_pred_embed))
