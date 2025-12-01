from pathlib import Path
import pandas as pd


def find_engineered_csv():
    # search up to 6 levels up for the data file (robust to where app is run)
    here = Path(__file__).resolve()
    for up in range(0, 7):
        try:
            candidate = here.parents[up] / "data" / "processed" / "engineered_features.csv"
        except IndexError:
            continue
        if candidate.exists():
            return candidate
    return None


def load_engineered_features():
    path = find_engineered_csv()
    if path is None:
        raise FileNotFoundError(
            "Could not locate data/processed/engineered_features.csv. Run from repo root or ensure data is present."
        )
    df = pd.read_csv(path, parse_dates=["gameDate"], infer_datetime_format=True)
    return df
