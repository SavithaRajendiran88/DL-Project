"""
model_pretrained_cnn.py
=======================
Transfer-learning approaches for wildlife movement anomaly detection.

The 1-D time-series windows are reshaped into 32×32×3 pseudo-images so that
pre-trained ImageNet backbones (ResNet50 and MobileNetV2) can extract
spatial-pattern features from the movement data.

Two stages:
  1. Feature extraction – backbone frozen, only the classification head is trained.
  2. Fine-tuning       – top 20 layers of MobileNetV2 unfrozen and retrained
                         with a much smaller learning rate.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2


# ── Image conversion helper ───────────────────────────────────────────────────

def windows_to_images(X: np.ndarray) -> np.ndarray:
    """
    Reshape a (N, 24, 4) window tensor into (N, 32, 32, 3) pseudo-images.

    The 96 feature values are zero-padded to 1024, reshaped to (32, 32, 1),
    and then replicated across 3 channels so ImageNet backbones accept the input.
    """
    N    = X.shape[0]
    flat = X.reshape(N, -1)                         # (N, 96)
    padded = np.zeros((N, 1024), dtype=np.float32)
    padded[:, :96] = flat
    img     = padded.reshape(N, 32, 32, 1)
    img_3ch = np.repeat(img, 3, axis=-1)            # (N, 32, 32, 3)
    return img_3ch


# ── Build helpers ─────────────────────────────────────────────────────────────

def build_resnet50_extractor() -> models.Sequential:
    """ResNet50 backbone (frozen) + Dense head."""
    resnet_base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3),
        pooling="avg",
    )
    resnet_base.trainable = False

    model = models.Sequential([
        resnet_base,
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="ResNet50_FeatureExtractor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    model.summary()
    return model


def build_mobilenetv2_extractor() -> tuple[models.Sequential, object]:
    """MobileNetV2 backbone (frozen) + Dense head. Returns (model, base)."""
    mobilenet_base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3),
        pooling="avg",
    )
    mobilenet_base.trainable = False

    model = models.Sequential([
        mobilenet_base,
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="MobileNetV2_FeatureExtractor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    model.summary()
    return model, mobilenet_base


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train_img: np.ndarray, y_train: np.ndarray,
    X_val_img:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    epochs: int = 20,
    batch_size: int = 64,
):
    """Generic fit call for feature-extraction stage. Returns history."""
    history = model.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ],
    )
    return history


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune_mobilenetv2(
    mobilenet_extractor,
    mobilenet_base,
    X_train_img: np.ndarray, y_train: np.ndarray,
    X_val_img:   np.ndarray, y_val:   np.ndarray,
    class_weights: dict,
    unfreeze_last_n: int = 20,
    epochs: int = 15,
    batch_size: int = 32,
):
    """
    Unfreeze the top `unfreeze_last_n` layers of MobileNetV2 and retrain
    with a small learning rate. Returns history.
    """
    mobilenet_base.trainable = True
    total_layers  = len(mobilenet_base.layers)
    freeze_until  = total_layers - unfreeze_last_n

    for layer in mobilenet_base.layers[:freeze_until]:
        layer.trainable = False
    for layer in mobilenet_base.layers[freeze_until:]:
        layer.trainable = True

    print(f"MobileNetV2 — total layers: {total_layers}")
    print(f"Frozen: {freeze_until}  |  Trainable: {total_layers - freeze_until}")

    mobilenet_extractor.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    history_ft = mobilenet_extractor.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ],
    )
    print("Fine-tuning complete.")
    return history_ft


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_feature_extraction_histories(
    history_resnet, history_mobilenet
) -> None:
    """2×2 grid: loss and accuracy for ResNet50 and MobileNetV2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Pretrained CNN Feature Extractors — Training History", fontsize=14
    )

    for ax, hist, name, col in zip(
        axes.flatten(),
        [history_resnet, history_resnet, history_mobilenet, history_mobilenet],
        ["ResNet50 Loss", "ResNet50 Accuracy",
         "MobileNetV2 Loss", "MobileNetV2 Accuracy"],
        ["loss", "accuracy", "loss", "accuracy"],
    ):
        ax.plot(hist.history[col],          label=f"Train {col.capitalize()}")
        ax.plot(hist.history[f"val_{col}"], label=f"Val {col.capitalize()}")
        ax.set_title(name)
        ax.set_xlabel("Epoch"); ax.set_ylabel(col.capitalize()); ax.legend()

    plt.tight_layout()
    plt.show()


def plot_fine_tuning_comparison(
    history_mobilenet, history_mobilenet_ft
) -> None:
    """Side-by-side val loss and val accuracy: before vs after fine-tuning."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_mobilenet.history["val_loss"],
             label="Before FT (val loss)", linestyle="--")
    plt.plot(history_mobilenet_ft.history["val_loss"],
             label="After FT (val loss)")
    plt.title("MobileNetV2 — Val Loss: Before vs After Fine-tuning")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_mobilenet.history["val_accuracy"],
             label="Before FT (val acc)", linestyle="--")
    plt.plot(history_mobilenet_ft.history["val_accuracy"],
             label="After FT (val acc)")
    plt.title("MobileNetV2 — Val Accuracy: Before vs After Fine-tuning")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader   import load_data
    from preprocessing import run_pipeline

    data = load_data()
    _, splits, _, class_weights, _ = run_pipeline(data)

    # Convert windows → pseudo-images
    X_train_img = windows_to_images(splits["X_train"])
    X_val_img   = windows_to_images(splits["X_val"])
    X_test_img  = windows_to_images(splits["X_test"])
    print("Image shapes — Train:", X_train_img.shape,
          " Val:", X_val_img.shape, " Test:", X_test_img.shape)

    # ResNet50 feature extraction
    resnet_extractor = build_resnet50_extractor()
    history_resnet   = train_model(
        resnet_extractor,
        X_train_img, splits["y_train"],
        X_val_img,   splits["y_val"],
        class_weights,
    )

    # MobileNetV2 feature extraction
    mobilenet_extractor, mobilenet_base = build_mobilenetv2_extractor()
    history_mobilenet = train_model(
        mobilenet_extractor,
        X_train_img, splits["y_train"],
        X_val_img,   splits["y_val"],
        class_weights,
    )

    plot_feature_extraction_histories(history_resnet, history_mobilenet)

    # Fine-tuning MobileNetV2
    history_mobilenet_ft = fine_tune_mobilenetv2(
        mobilenet_extractor, mobilenet_base,
        X_train_img, splits["y_train"],
        X_val_img,   splits["y_val"],
        class_weights,
    )
    plot_fine_tuning_comparison(history_mobilenet, history_mobilenet_ft)
