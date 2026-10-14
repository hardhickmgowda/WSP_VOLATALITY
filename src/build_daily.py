"""Step 3b: build a strictly CAUSAL daily panel for each stock.

Why a second dataset? The post-level dataset (build_dataset.py) pairs a post written
DURING trading day d with the move from the open of d to the open of d+1, so part of
that move has already happened when the post is written, and every post of one day shares
one label. Here one row = one trading day T and everything is known BEFORE the open of T:

  * Posts: every post dated on a calendar day BEFORE T and after the previous trading day's
    bucket, i.e. posts from the previous trading day, the weekend, or holidays. Timestamps in
    the dataset are UTC, so a UTC calendar day has finished before the next US open.
  * Coverage: the scraped dataset starts on 28 Jan 2021 and has scraping gaps (runs of days with
    no posts at all). Those days are dropped instead of being counted as "zero attention".
  * Prices: only data up to the CLOSE of the previous trading day (no peeking at the open of T).

Targets (what we try to predict for day T):
  ret_fwd      = Open(T+1) / Open(T) - 1          (buy at today's open, sell at the next open)
  direction    = 1 if ret_fwd > 0
  significant  = 1 if |ret_fwd| > median |move| of the previous 60 trading days (known at the open)
  log_abs_ret  = log(|ret_fwd| + 0.001)           (regression target for "how much will it move")

Run:  python -m src.build_daily
Output: data/processed/daily_<STOCK>.csv
"""
import ast

import numpy as np
import pandas as pd

from src import config

ALL_FEATURES = config.DAILY_PRICE_FEATURES + config.DAILY_WSB_FEATURES
TARGETS = ["ret_fwd", "abs_ret_fwd", "sig_thr"]


def _slog(x):
    """Signed log: keeps the sign but tames the huge squeeze-day returns (heavy tails)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _ns(series):
    """merge_asof needs identical datetime units on both sides (pandas 2 vs 3 differ)."""
    return pd.to_datetime(series).astype("datetime64[ns]")


def price_block(px: pd.DataFrame) -> pd.DataFrame:
    """Price features (known before the open of the row's date) and the targets."""
    px = px.sort_values("Date").reset_index(drop=True)
    o, h, l, c = px["Open"], px["High"], px["Low"], px["Close"]
    v = px["Volume"].astype(float)

    ret_cc = c.pct_change()                      # close-to-close return of each day
    rng = (h - l) / o                            # intraday range of each day

    f = pd.DataFrame({"Date": _ns(px["Date"])})
    # every feature is shifted by one day: row T only sees days up to T-1
    eps = 1e-3                                   # avoids log(0)
    f["ret_prev"] = _slog(ret_cc.shift(1))
    f["log_vol5"] = np.log(ret_cc.abs().rolling(5).mean().shift(1) + eps)
    f["log_vol20"] = np.log(ret_cc.abs().rolling(20).mean().shift(1) + eps)
    f["log_range1"] = np.log(rng.shift(1) + eps)
    f["log_range5"] = np.log(rng.rolling(5).mean().shift(1) + eps)
    f["mom5"] = _slog((c / c.shift(5) - 1).shift(1))
    f["vol_ratio"] = np.log(v.shift(1)) - np.log(v.rolling(20).mean().shift(2))

    # targets: open(T) -> open(T+1)
    f["ret_fwd"] = o.shift(-1) / o - 1
    f["abs_ret_fwd"] = f["ret_fwd"].abs()
    # Threshold for day T = median |move| realised by the open of T-1 (rows <= T-2). Lagging by two
    # rows (not one) keeps it independent of the opening price of T, which is not known when an
    # order is placed before the open.
    f["sig_thr"] = (f["abs_ret_fwd"].shift(2)
                    .rolling(config.SIG_WINDOW, min_periods=config.SIG_MIN_PERIODS).median())
    f["log_sig_thr"] = np.log(f["sig_thr"] + eps)   # the label's baseline, also a (past-only) feature
    return f


def bucketize(sub: pd.DataFrame, trading_days: pd.Series) -> pd.DataFrame:
    """Give every post the first trading day STRICTLY AFTER its date (weekend posts -> Monday,
    a post on trading day d -> the next trading day), so a post is always older than the open
    it is used for."""
    sub = sub.copy()
    sub["date"] = _ns(sub["date"])
    sub = sub.sort_values("date")
    cal = pd.DataFrame({"bucket": _ns(trading_days)}).sort_values("bucket")
    b = pd.merge_asof(sub, cal, left_on="date", right_on="bucket",
                      direction="forward", allow_exact_matches=False)   # strictly after
    return b.dropna(subset=["bucket"])


def corpus_coverage(posts: pd.DataFrame, trading_days: pd.Series) -> pd.Series:
    """Number of scraped posts (about ANY stock) per bucket, to detect scraping gaps."""
    b = bucketize(posts[posts["date"] >= pd.Timestamp(config.COVERAGE_START)], trading_days)
    return b.groupby("bucket").size().reindex(_ns(trading_days), fill_value=0)


def post_block(posts: pd.DataFrame, stock: str, trading_days: pd.Series) -> pd.DataFrame:
    """Aggregate the posts about `stock` per bucket."""
    mask = posts["mentions"].apply(lambda m: stock in m) & \
           (posts["date"] >= pd.Timestamp(config.COVERAGE_START))
    b = bucketize(posts[mask], trading_days)
    g = b.groupby("bucket").agg(
        n_posts=("id", "size"),
        comments=("total_comms_num", "sum"),
        score=("post_score", lambda s: s.clip(lower=0).sum()),
        mean_pos=("positive_sentiment", "mean"),
        mean_neg=("negative_sentiment", "mean"),
    )
    return g


def build_daily(posts: pd.DataFrame, px: pd.DataFrame, stock: str) -> pd.DataFrame:
    pf = price_block(px)
    g = post_block(posts, stock, pf["Date"])
    d = pf.set_index("Date").join(g)
    d["corpus_posts"] = corpus_coverage(posts, pf["Date"])

    # keep only days the scraped dataset actually covers (after the start date, no scraping gaps)
    d = d[d.index > pd.Timestamp(config.COVERAGE_START)]
    d = d[d["corpus_posts"] >= config.MIN_CORPUS_POSTS].copy()
    for col in ["n_posts", "comments", "score", "mean_pos", "mean_neg"]:
        d[col] = d[col].fillna(0.0)          # no post that day -> zero attention, neutral sentiment

    # WSB features (rolling windows only look backwards)
    d["log_posts"] = np.log1p(d["n_posts"])
    d["log_posts_ma5"] = np.log1p(d["n_posts"].rolling(5, min_periods=1).mean())
    baseline = d["n_posts"].shift(1).rolling(20, min_periods=3).mean()
    d["attn_shock"] = d["log_posts"] - np.log1p(baseline)
    d["log_comments"] = np.log1p(d["comments"])
    d["log_score"] = np.log1p(d["score"])

    d = d.iloc[config.WARMUP_DAYS:]
    d = d.dropna(subset=ALL_FEATURES + TARGETS).copy()

    d["direction"] = (d["ret_fwd"] > 0).astype(int)
    d["significant"] = (d["abs_ret_fwd"] > d["sig_thr"]).astype(int)
    d["log_abs_ret"] = np.log(d["abs_ret_fwd"] + 1e-3)
    d.index.name = "trade_date"
    keep = ALL_FEATURES + ["n_posts", "ret_fwd", "abs_ret_fwd", "sig_thr",
                           "direction", "significant", "log_abs_ret"]
    return d[keep].reset_index()


def main():
    posts = pd.read_csv(config.PROC_DIR / "posts_clean.csv", parse_dates=["date"])
    posts["mentions"] = posts["mentions"].apply(ast.literal_eval)

    for stock in config.MODEL_STOCKS:
        path = config.RAW_DIR / f"prices_{stock}.csv"
        if not path.exists():
            print(f"{stock}: no price file, run fetch_prices first")
            continue
        px = pd.read_csv(path, parse_dates=["Date"])
        if px["Date"].dt.tz is not None:
            px["Date"] = px["Date"].dt.tz_localize(None)
        ds = build_daily(posts, px, stock)
        out = config.PROC_DIR / f"daily_{stock}.csv"
        ds.to_csv(out, index=False)
        print(f"{stock}: {len(ds)} trading days {ds['trade_date'].min().date()} -> "
              f"{ds['trade_date'].max().date()} | days with posts={(ds['n_posts'] > 0).mean():.0%} | "
              f"up share={ds['direction'].mean():.2f} | significant share={ds['significant'].mean():.2f} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
