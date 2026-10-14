"""Step 1: load the WSB posts, clean the text, and add sentiment.

Run:  python -m src.preprocess
Output: data/processed/posts_clean.csv
"""
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src import config


def clean_text(text: str) -> str:
    """Same cleaning steps the paper describes (Section 3)."""
    text = str(text).lower()
    text = re.sub(r"@\w+", " ", text)                 # remove @handles
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove special characters / emojis
    text = re.sub(r"\b[a-z]\b", " ", text)            # remove single characters
    text = re.sub(r"\s+", " ", text).strip()          # many spaces -> one space
    return text


def find_mentions(text: str) -> list:
    """Return the list of stocks (from config.STOCKS) mentioned in a cleaned post."""
    words = set(text.split())
    return [s for s, info in config.STOCKS.items() if words & set(info["keywords"])]


def main():
    print(f"Reading {config.POSTS_CSV}")
    df = pd.read_csv(config.POSTS_CSV)
    print("Raw shape:", df.shape)

    # --- basic cleaning ---
    df = df.dropna(subset=["title", "timestamp"])          # title + time are required
    df["body"] = df["body"].fillna("")                      # many posts have no body
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["raw_text"] = df["title"].astype(str) + " " + df["body"].astype(str)
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # --- which stocks does each post mention? ---
    df["mentions"] = df["clean_text"].apply(find_mentions)

    # --- sentiment with VADER (works on the raw-ish text; VADER uses caps/punctuation) ---
    print("Running VADER sentiment (takes ~1 minute)...")
    sia = SentimentIntensityAnalyzer()
    scores = df["raw_text"].str.lower().apply(sia.polarity_scores)
    df["positive_sentiment"] = scores.apply(lambda s: s["pos"])
    df["neutral_sentiment"] = scores.apply(lambda s: s["neu"])
    df["negative_sentiment"] = scores.apply(lambda s: s["neg"])

    # --- the other features from the paper ---
    df["post_score"] = df["score"]
    df["total_comms_num"] = df["comms_num"]
    df["len_of_post"] = df["clean_text"].str.split().str.len()

    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROC_DIR / "posts_clean.csv"
    keep = ["id", "timestamp", "date", "title", "mentions"] + config.FEATURES
    df[keep].to_csv(out, index=False)
    print(f"Saved {len(df)} posts -> {out}")
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())


if __name__ == "__main__":
    main()
