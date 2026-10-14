"""Exploratory plots for the report/slides.

Run:  python -m src.eda
Outputs (results/): mentions.png (paper Figure 2), posts_per_day.png, sentiment.png
"""
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config


def main():
    df = pd.read_csv(config.PROC_DIR / "posts_clean.csv", parse_dates=["date"])
    df["mentions"] = df["mentions"].apply(ast.literal_eval)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # 1. how often each stock is mentioned (decides which stocks to model)
    counts = pd.Series({s: df["mentions"].apply(lambda m: s in m).sum() for s in config.STOCKS})
    counts.sort_values(ascending=False).plot.bar(color="teal", figsize=(7, 4))
    plt.title("Posts mentioning each stock"); plt.ylabel("Number of posts")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "mentions.png", dpi=150); plt.close()
    print(counts.sort_values(ascending=False).to_string())

    # 2. activity over time
    df.groupby("date").size().plot(figsize=(8, 3.5))
    plt.title("WSB posts per day"); plt.ylabel("Posts")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "posts_per_day.png", dpi=150); plt.close()

    # 3. sentiment distribution
    df[["positive_sentiment", "neutral_sentiment", "negative_sentiment"]].plot.hist(
        bins=30, alpha=.5, figsize=(7, 4))
    plt.title("VADER sentiment of posts")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "sentiment.png", dpi=150); plt.close()

    print(df[config.FEATURES].describe().round(2).to_string())
    print(f"Saved plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
