import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_PATH = BASE / "data" / "raw" / "transactions.csv"
PROCESSED_PATH = BASE / "data" / "processed" / "features.csv"


def load_data(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def engineer_features(df):
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(22, 6).astype(int)

    # User-level aggregates
    user_stats = df.groupby("user_id")["amount"].agg(
        user_avg_amount="mean",
        user_std_amount="std",
        user_tx_count="count"
    ).reset_index()
    df = df.merge(user_stats, on="user_id", how="left")
    df["user_std_amount"] = df["user_std_amount"].fillna(0)

    # Amount deviation from user average
    df["amount_deviation"] = (df["amount"] - df["user_avg_amount"]) / (df["user_std_amount"] + 1)

    # Transaction frequency per user per day
    df["date"] = df["timestamp"].dt.date
    daily_freq = df.groupby(["user_id", "date"]).size().reset_index(name="daily_tx_count")
    df = df.merge(daily_freq, on=["user_id", "date"], how="left")

    # Time gap between consecutive transactions per user
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["time_gap_minutes"] = (
        df["timestamp"] - df["prev_timestamp"]
    ).dt.total_seconds().div(60).fillna(9999)

    # Device and location consistency per user
    user_devices = df.groupby("user_id")["device_info"].nunique().reset_index(name="device_variety")
    user_locations = df.groupby("user_id")["location"].nunique().reset_index(name="location_variety")
    df = df.merge(user_devices, on="user_id", how="left")
    df = df.merge(user_locations, on="user_id", how="left")

    # Encode categorical
    df["device_encoded"] = df["device_info"].map(
        {"Android": 0, "iOS": 1, "Web": 2, "Unknown": 3}
    ).fillna(3)
    df["location_encoded"] = df["location"].astype("category").cat.codes

    # Drop helper columns
    df = df.drop(columns=["prev_timestamp", "date"])

    return df


def save_features(df, path=PROCESSED_PATH):
    df.to_csv(path, index=False)
    print(f"Saved processed features: {path} ({len(df)} rows)")


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    save_features(df)
    print(df[["transaction_id", "amount", "amount_deviation", "daily_tx_count",
               "time_gap_minutes", "device_variety", "is_fraud"]].head(10))
