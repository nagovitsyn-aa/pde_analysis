import pandas as pd


def normalize_parameters(df: pd.DataFrame, ndigits: int = 5) -> pd.DataFrame:
    df = df.copy()

    if "u" in df.columns:
        df["u"] = df["u"].round(ndigits)

    if "lambda_" in df.columns:
        df["lambda_"] = df["lambda_"].round(ndigits)

    return df


def dropna_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.dropna(subset=[metric])