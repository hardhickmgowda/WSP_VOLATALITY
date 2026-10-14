"""Step 2: download daily stock prices from Yahoo Finance (yfinance).

Run:  python -m src.fetch_prices
Output: data/raw/prices_<STOCK>.csv  (Date, Open, Close, ...)
"""
import pandas as pd
import yfinance as yf

from src import config


def main():
    posts = pd.read_csv(config.PROC_DIR / "posts_clean.csv", parse_dates=["date"])
    # a few extra days on each side so every post has a "next trading day"
    start = (posts["date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (posts["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    for stock in config.MODEL_STOCKS:
        ticker = config.STOCKS[stock]["yf"]
        print(f"Downloading {ticker} ({stock}) {start} -> {end}")
        px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(px.columns, pd.MultiIndex):        # newer yfinance returns 2-level columns
            px.columns = px.columns.get_level_values(0)
        if px.empty:
            print(f"  WARNING: no data for {ticker}")
            continue
        px = px.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        out = config.RAW_DIR / f"prices_{stock}.csv"
        px.to_csv(out, index=False)
        print(f"  saved {len(px)} trading days -> {out}")


if __name__ == "__main__":
    main()
