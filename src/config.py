"""Central settings for the project. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# The posts CSV (downloaded by src/download_data.py from OpenML dataset 43373)
POSTS_CSV = RAW_DIR / "reddit_wsb.csv"

# Stocks to study. Key = name used in the project, value = list of words that
# count as a "mention" of that stock in a post, and the Yahoo Finance ticker.
# Note: Facebook's ticker changed from FB to META in 2022; Yahoo keeps the
# old history under META.
STOCKS = {
    "GME":  {"keywords": ["gme", "gamestop"],          "yf": "GME"},
    "AMC":  {"keywords": ["amc"],                      "yf": "AMC"},
    "TSLA": {"keywords": ["tsla", "tesla"],            "yf": "TSLA"},
    "FB":   {"keywords": ["fb", "facebook"],           "yf": "META"},
    # extra tickers only used in the "mentions" histogram (Figure 2)
    "AAPL": {"keywords": ["aapl", "apple"],            "yf": "AAPL"},
    "MSFT": {"keywords": ["msft", "microsoft"],        "yf": "MSFT"},
    "GOOGL": {"keywords": ["googl", "google"],         "yf": "GOOGL"},
}
MODEL_STOCKS = ["GME", "AMC", "TSLA", "FB"]   # the ones we train models for

# "Significant shift" label: a move bigger than the stock's median daily move
# (computed per stock in build_dataset.py)

FEATURES = [
    "post_score",
    "total_comms_num",
    "len_of_post",
    "positive_sentiment",
    "neutral_sentiment",
    "negative_sentiment",
]  # the bias column (all 1s) is added inside the logistic-regression code

SEED = 42


# ---------------------------------------------------------------------------
# Daily (causal) pipeline: src/build_daily.py and src/walk_forward.py
# ---------------------------------------------------------------------------
# The scraped dataset is essentially empty before 28 Jan 2021 (1 post in Sep-Jan),
# so zero posts before this date means "missing data", not "no attention".
COVERAGE_START = "2021-01-28"
# Days on which the WHOLE subreddit has fewer than this many scraped posts are treated as
# scraping gaps (the dataset has runs of days with 0 posts, e.g. 22 Jul - 2 Aug 2021) and
# are dropped; a gap is missing data, not "nobody talked about the stock".
MIN_CORPUS_POSTS = 20
WARMUP_DAYS = 5            # first trading days dropped so rolling features have history

# "Significant move" = |next open-to-open move| above the MEDIAN of the previous
# 60 trading days (only past information, unlike a full-sample median).
SIG_WINDOW = 60
SIG_MIN_PERIODS = 20

# Walk-forward evaluation: train on days [0, t), predict the next DAILY_STEP days,
# then move forward and refit. The first DAILY_MIN_TRAIN days are never predicted.
DAILY_MIN_TRAIN = 40
DAILY_STEP = 5

TRANSACTION_COST = 0.001   # 10 bps every time the position changes (enter or exit)
N_PERM = 500               # label permutations for the AUC p-value
N_BOOT = 500               # block-bootstrap resamples for the AUC-difference CI
BOOT_BLOCK = 5             # block length (trading days) keeps autocorrelation in the resamples

# Features known BEFORE the open of the day being predicted (see build_daily.py)
DAILY_PRICE_FEATURES = [
    "log_vol5", "log_vol20",         # log of recent realised volatility (mean |close-to-close return|)
    "log_range1", "log_range5",      # log of the intraday high-low range of the previous day(s)
    "ret_prev", "mom5",              # last day's return, 5-day momentum (signed-log scaled)
    "vol_ratio",                     # previous day's volume vs its 20-day average (log ratio)
    "log_sig_thr",                   # log of the typical recent move that the "significant" label is
                                     # measured against (known before the open, so it may be a feature)
]
DAILY_WSB_FEATURES = [
    "log_posts", "log_posts_ma5",    # how many posts mention the stock
    "attn_shock",                    # today's attention vs the trailing 20-day average
    "log_comments", "log_score",
    "mean_pos", "mean_neg",          # average VADER sentiment
]
DAILY_FEATURE_SETS = {
    "price_only": DAILY_PRICE_FEATURES,
    "wsb_only": DAILY_WSB_FEATURES,
    "price_plus_wsb": DAILY_PRICE_FEATURES + DAILY_WSB_FEATURES,
}
