"""Automated checks for the daily pipeline.  Run from the project root:  python -m pytest -q

The most important test is test_features_are_causal: it rewrites everything that happens on or
after day T (prices AND posts) and checks that the features of day T do not change.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import build_daily, config
from src.models import NewtonLogisticRegression
from src.walk_forward import backtest, walk_forward

FEATS = config.DAILY_PRICE_FEATURES + config.DAILY_WSB_FEATURES


# ------------------------------------------------------------------ synthetic data helpers
def make_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    o = 20 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    c = o * np.exp(rng.normal(0, 0.02, n))
    h, l = np.maximum(o, c) * 1.02, np.minimum(o, c) * 0.98
    return pd.DataFrame({"Date": dates, "Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": rng.integers(1e6, 5e6, n)})


def make_posts(px, seed=1):
    """~30 posts per calendar day (weekends included), all mentioning GME."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(px["Date"].min(), px["Date"].max() + pd.Timedelta(days=3))
    rows = []
    for d in days:
        for k in range(30):
            rows.append({"id": f"{d.date()}_{k}", "date": d, "mentions": ["GME"],
                         "post_score": int(rng.integers(0, 500)),
                         "total_comms_num": int(rng.integers(0, 100)),
                         "positive_sentiment": rng.uniform(0, .3), "negative_sentiment": rng.uniform(0, .3)})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def synthetic_config(monkeypatch):
    monkeypatch.setattr(config, "COVERAGE_START", "2021-01-04")
    monkeypatch.setattr(config, "MIN_CORPUS_POSTS", 20)


# ------------------------------------------------------------------ no look-ahead
def test_posts_are_assigned_to_the_next_trading_day():
    days = pd.Series(pd.to_datetime(["2021-02-03", "2021-02-04", "2021-02-05", "2021-02-08"]))  # Wed..Mon
    posts = pd.DataFrame({"id": [1, 2, 3, 4],
                          "date": pd.to_datetime(["2021-02-03", "2021-02-05", "2021-02-06", "2021-02-07"])})
    b = build_daily.bucketize(posts, days).set_index("id")["bucket"]
    assert b[1] == pd.Timestamp("2021-02-04")     # Wednesday post -> Thursday (never same day)
    assert b[2] == pd.Timestamp("2021-02-08")     # Friday post    -> Monday
    assert b[3] == pd.Timestamp("2021-02-08")     # Saturday post  -> Monday
    assert b[4] == pd.Timestamp("2021-02-08")     # Sunday post    -> Monday


def test_features_are_causal():
    px, posts = make_prices(), None
    posts = make_posts(px)
    base = build_daily.build_daily(posts, px, "GME")
    assert len(base) > 60

    # many days: a median-based feature only reacts to a leak on some of them
    for k in range(10, 90, 6):
        t = base["trade_date"].iloc[k]
        px2, posts2 = px.copy(), posts.copy()
        future_px = px2["Date"] >= t                       # everything on/after day T changes
        px2.loc[future_px, ["Open", "High", "Low", "Close"]] *= 7.3
        px2.loc[future_px, "Volume"] *= 11
        future_posts = posts2["date"] >= t
        posts2.loc[future_posts, ["post_score", "total_comms_num"]] *= 9
        posts2.loc[future_posts, "positive_sentiment"] = 0.99
        changed = build_daily.build_daily(posts2, px2, "GME")

        a = base.set_index("trade_date").loc[t, FEATS].astype(float)
        b = changed.set_index("trade_date").loc[t, FEATS].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False)
        # the earlier rows must be untouched too
        pd.testing.assert_frame_equal(base.iloc[:k][FEATS].reset_index(drop=True),
                                      changed.iloc[:k][FEATS].reset_index(drop=True))


def test_label_threshold_does_not_use_todays_open():
    """sig_thr (used for the label AND as a feature) must not depend on Open(T)."""
    px = make_prices()
    posts = make_posts(px)
    base = build_daily.build_daily(posts, px, "GME").set_index("trade_date")
    for t in base.index[5:-5]:                              # every day: the median is robust, so a
        px2 = px.copy()                                     # leak only shows on some of them
        px2.loc[px2["Date"] == t, "Open"] *= 3              # change only the opening price of T
        changed = build_daily.build_daily(posts, px2, "GME").set_index("trade_date")
        assert changed.loc[t, "sig_thr"] == pytest.approx(base.loc[t, "sig_thr"])


def test_scraping_gaps_are_dropped_not_zero_filled():
    px = make_prices()
    posts = make_posts(px)
    gap_start, gap_end = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-05")
    posts = posts[~posts["date"].between(gap_start, gap_end)]
    out = build_daily.build_daily(posts, px, "GME")
    # buckets whose posts all fall in the gap have no data -> no row
    assert not out["trade_date"].between(pd.Timestamp("2021-03-02"), pd.Timestamp("2021-03-05")).any()


def test_walk_forward_only_trains_on_the_past():
    class Spy:
        """Predicts the number of rows it was trained on, so we can see what each block used."""
        def fit(self, X, y):
            self.n = len(X)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), self.n / 1000)])

    n, mt, step = 60, 20, 5
    X, y = np.random.default_rng(0).normal(size=(n, 3)), np.tile([0, 1], n // 2)
    out = walk_forward(X, y, Spy, "clf", mt, step)
    assert np.isnan(out[:mt]).all() and not np.isnan(out[mt:]).any()
    for start in range(mt, n, step):
        assert out[start:start + step] == pytest.approx(start / 1000)   # trained on rows [0, start) only


def test_no_signal_in_random_features():
    from src.models import make_classifier
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(300, 6)), rng.integers(0, 2, 300)
    s = walk_forward(X, y, lambda: make_classifier("logreg"), "clf", 60, 5)
    m = ~np.isnan(s)
    assert 0.40 < roc_auc_score(y[m], s[m]) < 0.60


def test_real_signal_is_detected():
    from src.models import make_classifier
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    s = walk_forward(X, y, lambda: make_classifier("logreg"), "clf", 60, 5)
    m = ~np.isnan(s)
    assert roc_auc_score(y[m], s[m]) > 0.9


# ------------------------------------------------------------------ models and backtest
def test_newton_logistic_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X @ np.array([1.0, -0.5, 0.3, 0.0]) + rng.normal(size=400) > 0).astype(int)
    l2 = 0.1
    ours = NewtonLogisticRegression(max_iter=100, tol=1e-12, l2=l2, penalize_bias=False).fit(X, y)
    sk = LogisticRegression(C=1 / (len(X) * l2), tol=1e-10, max_iter=1000).fit(X, y)
    np.testing.assert_allclose(ours.theta[:-1], sk.coef_[0], atol=1e-3)
    assert ours.theta[-1] == pytest.approx(sk.intercept_[0], abs=1e-3)


def test_balanced_class_weight_predicts_more_of_the_rare_class():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] + rng.normal(size=600) > 1.5).astype(int)           # rare positives
    plain = NewtonLogisticRegression(l2=0.01).fit(X, y).predict(X).sum()
    balanced = NewtonLogisticRegression(l2=0.01, class_weight="balanced").fit(X, y).predict(X).sum()
    assert balanced > plain


def test_backtest_costs_and_cash():
    ret = np.array([0.10, -0.05, 0.02])
    res, _ = backtest(ret, np.ones(3), cost=0.001)       # long all days: one entry cost only
    expected = (1 + 0.10 - 0.001) * 0.95 * 1.02 - 1
    assert res["total_return"] == pytest.approx(expected, abs=1e-3)
    res0, _ = backtest(ret, np.zeros(3), cost=0.001)     # always cash: nothing happens
    assert res0["total_return"] == 0 and res0["n_trades"] == 0
