"""Step 3: join posts with prices and create labels -> one dataset per stock.

Label "direction"   : 1 if Open(next trading day) > Open(day of post), else 0
Label "significant" : 1 if |Open(next)/Open(today) - 1| > that stock's MEDIAN daily move
                      (so ~half the trading days count as "significant" for every stock;
                      a fixed 3% would mark 90%+ of GME/AMC days as significant)

Posts made on a weekend/holiday are assigned to the next trading day.

Run:  python -m src.build_dataset
Output: data/processed/dataset_<STOCK>.csv
"""
import ast

import pandas as pd

from src import config


def build_for_stock(posts: pd.DataFrame, stock: str) -> pd.DataFrame:
    px = pd.read_csv(config.RAW_DIR / f"prices_{stock}.csv", parse_dates=["Date"])
    px = px.sort_values("Date").reset_index(drop=True)
    px["Date"] = px["Date"].dt.tz_localize(None) if px["Date"].dt.tz is not None else px["Date"]
    px["next_open"] = px["Open"].shift(-1)
    px["ret_next_open"] = px["next_open"] / px["Open"] - 1
    px = px.dropna(subset=["next_open"])
    threshold = px["ret_next_open"].abs().median()   # typical daily move of THIS stock

    sub = posts[posts["mentions"].apply(lambda m: stock in m)].copy()
    sub = sub.sort_values("date")

    # map each post to the first trading day on/after its date
    merged = pd.merge_asof(sub, px[["Date", "Open", "next_open", "ret_next_open"]],
                           left_on="date", right_on="Date", direction="forward")
    merged = merged.dropna(subset=["Date"])

    merged["label_direction"] = (merged["next_open"] > merged["Open"]).astype(int)
    merged["label_significant"] = (merged["ret_next_open"].abs() > threshold).astype(int)
    merged = merged.rename(columns={"Date": "trade_date"})
    merged["sig_threshold"] = threshold
    return merged


def main():
    posts = pd.read_csv(config.PROC_DIR / "posts_clean.csv", parse_dates=["date"])
    posts["mentions"] = posts["mentions"].apply(ast.literal_eval)   # stored as text "['GME']"

    for stock in config.MODEL_STOCKS:
        try:
            ds = build_for_stock(posts, stock)
        except FileNotFoundError:
            print(f"{stock}: no price file, run fetch_prices first")
            continue
        out = config.PROC_DIR / f"dataset_{stock}.csv"
        ds.to_csv(out, index=False)
        print(f"{stock}: {len(ds)} posts over {ds['trade_date'].nunique()} trading days | "
              f"up-days share={ds['label_direction'].mean():.2f} | "
              f"significant (>{ds['sig_threshold'].iloc[0]:.1%}) share={ds['label_significant'].mean():.2f} -> {out.name}")


if __name__ == "__main__":
    main()
