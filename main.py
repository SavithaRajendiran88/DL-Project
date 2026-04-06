"""
main.py
=======
Orchestration script — runs the full Wildlife Anomaly Detection pipeline
by wiring all modules together in the correct order.

Usage (in Google Colab or locally):
    python main.py
"""

# ── 1. Data loading ────────────────────────────────────────────────────────────
from data_loader import load_data

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
from preprocessing import run_pipeline

# ── 3. Models ─────────────────────────────────────────────────────────────────
from model_mlp             import train_mlp,            predict_mlp,            plot_history as plot_mlp
from model_cnn             import train_cnn,            predict_cnn,            plot_history as plot_cnn
from model_rnn             import train_rnn,            predict_rnn,            plot_history as plot_rnn
from model_lstm            import train_lstm,           predict_lstm,           plot_history as plot_lstm
from model_gru             import train_gru,            predict_gru,            plot_history as plot_gru, hyperparameter_search, plot_hp_results
from model_pretrained_cnn  import (windows_to_images, build_resnet50_extractor,
                                    build_mobilenetv2_extractor, train_model,
                                    fine_tune_mobilenetv2,
                                    plot_feature_extraction_histories,
                                    plot_fine_tuning_comparison)
from model_attention_lstm  import train_attention_lstm, predict_attention_lstm, plot_history as plot_attn
from model_embedding_lstm  import (build_elephant_vocab, make_split_id_arrays,
                                    train_embedding_lstm, predict_embedding_lstm,
                                    plot_history as plot_embed)

# ── 4. Evaluation ──────────────────────────────────────────────────────────────
from evaluation import run_evaluation


def main():
    # ── Step 1: Load raw data ──────────────────────────────────────────────────
    print("\n=== Loading data ===")
    data = load_data()

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    print("\n=== Preprocessing ===")
    df_model, splits, window_elephants, class_weights, scaler = run_pipeline(data)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    # ── Step 3a: MLP ──────────────────────────────────────────────────────────
    print("\n=== Training MLP ===")
    mlp, history_mlp, _ = train_mlp(X_train, y_train, X_val, y_val, class_weights)
    plot_mlp(history_mlp)
    y_prob_mlp, y_pred_mlp = predict_mlp(mlp, X_test)

    # ── Step 3b: CNN ──────────────────────────────────────────────────────────
    print("\n=== Training CNN ===")
    cnn, history_cnn = train_cnn(X_train, y_train, X_val, y_val, class_weights)
    plot_cnn(history_cnn)
    y_prob_cnn, y_pred_cnn = predict_cnn(cnn, X_test)

    # ── Step 3c: RNN ──────────────────────────────────────────────────────────
    print("\n=== Training RNN ===")
    rnn_model, history_rnn = train_rnn(X_train, y_train, X_val, y_val, class_weights)
    plot_rnn(history_rnn)
    y_prob_rnn, y_pred_rnn = predict_rnn(rnn_model, X_test)

    # ── Step 3d: LSTM ─────────────────────────────────────────────────────────
    print("\n=== Training LSTM ===")
    lstm_model, history_lstm = train_lstm(X_train, y_train, X_val, y_val, class_weights)
    plot_lstm(history_lstm)
    y_prob_lstm, y_pred_lstm = predict_lstm(lstm_model, X_test)

    # ── Step 3e: GRU + hyperparameter search ──────────────────────────────────
    print("\n=== Training GRU ===")
    gru_model, history_gru = train_gru(X_train, y_train, X_val, y_val, class_weights)
    plot_gru(history_gru)
    y_prob_gru, y_pred_gru = predict_gru(gru_model, X_test)

    print("\n=== GRU Hyperparameter Search ===")
    hp_df = hyperparameter_search(X_train, y_train, X_val, y_val, class_weights)
    plot_hp_results(hp_df)

    # ── Step 3f: Pretrained CNNs ──────────────────────────────────────────────
    print("\n=== Pretrained CNN Feature Extraction ===")
    X_train_img = windows_to_images(X_train)
    X_val_img   = windows_to_images(X_val)
    X_test_img  = windows_to_images(X_test)

    resnet_extractor = build_resnet50_extractor()
    history_resnet   = train_model(resnet_extractor, X_train_img, y_train,
                                   X_val_img, y_val, class_weights)

    mobilenet_extractor, mobilenet_base = build_mobilenetv2_extractor()
    history_mobilenet = train_model(mobilenet_extractor, X_train_img, y_train,
                                    X_val_img, y_val, class_weights)

    plot_feature_extraction_histories(history_resnet, history_mobilenet)

    print("\n=== Fine-tuning MobileNetV2 ===")
    history_mobilenet_ft = fine_tune_mobilenetv2(
        mobilenet_extractor, mobilenet_base,
        X_train_img, y_train, X_val_img, y_val, class_weights,
    )
    plot_fine_tuning_comparison(history_mobilenet, history_mobilenet_ft)

    # ── Step 3g: Attention-LSTM ───────────────────────────────────────────────
    print("\n=== Training Attention-LSTM ===")
    attn_model, history_attn = train_attention_lstm(
        X_train, y_train, X_val, y_val, class_weights
    )
    plot_attn(history_attn)
    y_prob_attn, y_pred_attn = predict_attention_lstm(attn_model, X_test)

    # ── Step 3h: LSTM + Embedding ─────────────────────────────────────────────
    print("\n=== Training LSTM + Embedding ===")
    elephant_to_idx = build_elephant_vocab(df_model)
    n_elephants     = len(elephant_to_idx)

    eid_train, eid_val, eid_test = make_split_id_arrays(
        window_elephants,
        elephant_to_idx,
        splits["train_mask"],
        splits["val_mask"],
        splits["test_mask"],
    )

    embed_model, history_embed = train_embedding_lstm(
        X_train, eid_train, y_train,
        X_val,   eid_val,   y_val,
        n_elephants=n_elephants,
        class_weights=class_weights,
    )
    plot_embed(history_embed)
    y_prob_embed, y_pred_embed = predict_embedding_lstm(embed_model, X_test, eid_test)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    print("\n=== Evaluation ===")
    model_registry = {
        "MLP":            (y_pred_mlp.ravel(),   y_prob_mlp.ravel()),
        "CNN":            (y_pred_cnn.ravel(),   y_prob_cnn.ravel()),
        "RNN":            (y_pred_rnn.ravel(),   y_prob_rnn.ravel()),
        "LSTM":           (y_pred_lstm.ravel(),  y_prob_lstm.ravel()),
        "GRU":            (y_pred_gru.ravel(),   y_prob_gru.ravel()),
        "Attention-LSTM": (y_pred_attn.ravel(),  y_prob_attn.ravel()),
        "LSTM+Embedding": (y_pred_embed.ravel(), y_prob_embed.ravel()),
    }

    metrics_df = run_evaluation(model_registry, y_test)
    return metrics_df


if __name__ == "__main__":
    main()
