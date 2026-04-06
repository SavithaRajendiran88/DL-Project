"""
preprocessing.py
================
Data cleaning, feature engineering, windowing, and train/val/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ── Constants ─────────────────────────────────────────────────────────────────
SELECTED_ELEPHANTS = ["LA14", "LA11", "LA13", "LA12"]
TRAIN_ELEPHANTS    = ["LA11", "LA12"]
VAL_ELEPHANTS      = ["LA13"]
TEST_ELEPHANTS     = ["LA14"]
DATE_START         = "2010-01-01"
DATE_END           = "2010-12-31"
WINDOW             = 24
ANOMALY_THRESHOLD  = 0.2   # fraction of anomalous steps in a window → label=1
SPEED_PERCENTILE   = 0.95  # top 5 % speed is "anomalous"
FEATURES           = ["location-lat", "location-long", "speed_kmh", "distance_km"]


# ── Helper functions ──────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) between two GPS points on Earth's surface."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Pipeline steps ────────────────────────────────────────────────────────────

def select_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the four relevant columns and parse timestamps."""
    df = data[["timestamp", "location-long", "location-lat",
               "individual-local-identifier"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the four most-tracked elephants within the 2010 calendar year."""
    mask = (
        df["individual-local-identifier"].isin(SELECTED_ELEPHANTS)
        & (df["timestamp"] >= DATE_START)
        & (df["timestamp"] <= DATE_END)
    )
    df_filtered = df[mask].copy()
    print("Filtered rows:", len(df_filtered))
    return df_filtered


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distance, time-diff, speed, and anomaly label per elephant."""
    df = df.sort_values(["individual-local-identifier", "timestamp"]).copy()

    # Lagged positions for distance calculation
    df["lat_prev"] = df.groupby("individual-local-identifier")["location-lat"].shift()
    df["lon_prev"] = df.groupby("individual-local-identifier")["location-long"].shift()

    df["distance_km"] = haversine(
        df["lat_prev"], df["lon_prev"],
        df["location-lat"], df["location-long"]
    )

    # Time diff in hours
    df["time_diff_hr"] = (
        df.groupby("individual-local-identifier")["timestamp"]
        .diff()
        .dt.total_seconds() / 3600
    )

    df["speed_kmh"] = df["distance_km"] / df["time_diff_hr"]
    df = df.dropna()

    # Binary anomaly label: top-5 % speed → 1
    threshold = df["speed_kmh"].quantile(SPEED_PERCENTILE)
    df["label"] = (df["speed_kmh"] > threshold).astype(int)

    print("Label distribution (normalised):")
    print(df["label"].value_counts(normalize=True))
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale the four input features in-place; return scaler."""
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df, scaler


def make_windows(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a WINDOW-length window over each elephant's time series.

    Returns
    -------
    X_windows       : (N, WINDOW, n_features)
    y_windows       : (N,)  — 1 if >20 % of steps in the window are anomalous
    window_elephants: (N,)  — elephant ID for each window
    """
    X_windows, y_windows, window_elephants = [], [], []

    for eid in df["individual-local-identifier"].unique():
        df_e = df[df["individual-local-identifier"] == eid]
        X_e  = df_e[FEATURES].values
        y_e  = df_e["label"].values

        for i in range(len(X_e) - WINDOW):
            window_labels = y_e[i : i + WINDOW]
            label = 1 if window_labels.mean() > ANOMALY_THRESHOLD else 0
            X_windows.append(X_e[i : i + WINDOW])
            y_windows.append(label)
            window_elephants.append(eid)

    X_windows        = np.array(X_windows)
    y_windows        = np.array(y_windows)
    window_elephants = np.array(window_elephants)

    print("Window tensor shape:", X_windows.shape)
    return X_windows, y_windows, window_elephants


def split_by_elephant(
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    window_elephants: np.ndarray,
) -> dict:
    """
    Elephant-stratified train / val / test split.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    train_mask = np.isin(window_elephants, TRAIN_ELEPHANTS)
    val_mask   = np.isin(window_elephants, VAL_ELEPHANTS)
    test_mask  = np.isin(window_elephants, TEST_ELEPHANTS)

    splits = dict(
        X_train = X_windows[train_mask],
        y_train = y_windows[train_mask],
        X_val   = X_windows[val_mask],
        y_val   = y_windows[val_mask],
        X_test  = X_windows[test_mask],
        y_test  = y_windows[test_mask],
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )

    for split in ("Train", "Val", "Test"):
        key = split.lower()
        print(f"{split}: {splits[f'X_{key}'].shape}  "
              f"Anomaly ratio: {splits[f'y_{key}'].mean():.3f}")
    return splits


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Return class-weight dict suitable for Keras `class_weight=` argument."""
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = dict(enumerate(cw))
    print("Class weights:", class_weights)
    return class_weights


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(data: pd.DataFrame):
    """
    Execute all preprocessing steps end-to-end.

    Returns
    -------
    df_model         : scaled, feature-engineered DataFrame
    splits           : dict of X/y train/val/test arrays
    window_elephants : per-window elephant IDs (full, unmasked)
    class_weights    : dict for Keras training
    scaler           : fitted StandardScaler
    """
    df = select_columns(data)
    df = filter_data(df)
    df = df[["timestamp", "location-lat", "location-long",
             "individual-local-identifier"]].copy()
    df = engineer_features(df)
    df, scaler = scale_features(df)

    X_windows, y_windows, window_elephants = make_windows(df)
    splits       = split_by_elephant(X_windows, y_windows, window_elephants)
    class_weights = compute_class_weights(splits["y_train"])

    return df, splits, window_elephants, class_weights, scaler


if __name__ == "__main__":
    from data_loader import load_data
    data = load_data()
    df_model, splits, window_elephants, class_weights, scaler = run_pipeline(data)
