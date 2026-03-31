import pandas as pd
from pathlib import Path
from typing import Tuple
#Path anpassen
RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "trustpilot_reviews_production"
PROCESSED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "reviews_clean.csv"

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Trustpilot data.
    Returns a DataFrame.
    """
    # Anpassen je nach Format (CSV/JSON)
    if RAW_PATH.suffix == ".csv":
        df = pd.read_csv(RAW_PATH)
    else:
        df = pd.read_json(RAW_PATH)
    return df

def load_processed_data() -> pd.DataFrame:
    """
    Load already cleaned/processed data.
    """
    return pd.read_csv(PROCESSED_PATH)

def get_data(use_processed: bool = True) -> pd.DataFrame:
    """
    Utility function to get either raw or processed data.
    """
    if use_processed:
        return load_processed_data()
    else:
        return load_raw_data()

if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(f"Dataset size: {df.shape}")