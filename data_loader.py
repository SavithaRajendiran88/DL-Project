"""
data_loader.py
==============
Loading and initial exploration of the Elephant GPS dataset.
"""

import pandas as pd


DATA_PATH = "African elephants in Etosha National Park (data from Tsalyuk et al. 2018) (5).csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV from Google Drive."""
    data = pd.read_csv(path, engine="python")
    return data


def explore_data(data: pd.DataFrame) -> None:
    """Print basic exploration stats for the raw dataset."""
    print("Shape:", data.shape)
    print("\nHead:")
    print(data.head())
    print("\nDescribe:")
    print(data.describe())
    print("\nInfo:")
    data.info()
    print("\nMissing values:")
    print(data.isna().sum())

    # Column-level value counts
    for col in [
        "visible",
        "sensor-type",
        "individual-taxon-canonical-name",
        "tag-local-identifier",
        "individual-local-identifier",
        "study-name",
    ]:
        print(f"\n{col} value counts:")
        print(data[col].value_counts())


if __name__ == "__main__":
    data = load_data()
    explore_data(data)
