"""Step 6: honest, causal evaluation on the daily panel (build_daily.py).

Differences from the post-level experiments in train.py:
  * Walk-forward (expanding window): train on days [0, t), predict the next 5 days, refit, repeat.
    Every prediction is out-of-sample and never uses the future. Scaling is fitted on the training
    window only.
  * Three feature sets, to answer "does WSB add anything?":
        price_only       past volatility / range / momentum / volume (no WSB at all)
        wsb_only         post counts, attention shock, comments, score, sentiment
        price_plus_wsb   both
    Past volatility is by far the best predictor of future volatility, so a WSB effect only counts
    if it improves on the price-only model.
  * Three tasks: "significant" move (classification), "direction" (classification), and
    "how much" (regression of log |move|).
  * Metrics for imbalanced data (balanced accuracy, F1, MCC, ROC-AUC, PR-AUC), a label-permutation
    p-value for AUC, and a block-bootstrap confidence interval for the AUC gain from adding WSB.
  * A trading backtest with transaction costs, Sharpe ratio and drawdown.
  * Permutation importance of each feature.

Run:  python -m src.walk_forward            # all models (about 1-3 minutes)
      python -m src.walk_forward --fast     # skip the slow models (random forest, MLP)
Outputs (results/): daily_classification.csv, daily_regression.csv, daily_incremental_value.csv,
                    daily_strategy.csv, daily_importance.csv and the daily_*.png plots.
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import (CLASSIFIERS, REGRESSORS, make_classifier, make_regressor,
                        positive_proba)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PRIMARY = "logreg"          # the pre-declared main model: our own Newton's-method logistic regression
TASKS = {"significant": "significant", "direction": "direction"}


# --------------------------------------------------------------------------- walk-forward core
def walk_forward(X, y, make_model, task, min_train, step):
    """Out-of-sample predictions for rows min_train..n-1 (NaN before that).

    task "clf": returns P(y=1);  task "reg": returns the predicted value.
    Training window is always rows [0, start): nothing from the test block or later is used.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    out = np.full(n, np.nan)
    for start in range(min_train, n, step):
        stop = min(start + step, n)
        scaler = StandardScaler().fit(X[:start])
        Xtr, Xte = scaler.transform(X[:start]), scaler.transform(X[start:stop])
        ytr = y[:start]
        if task == "clf":
            if len(np.unique(ytr)) < 2:              # only one class seen so far
                out[start:stop] = ytr.mean()
                continue
            out[start:stop] = positive_proba(make_model().fit(Xtr, ytr.astype(int)), Xte)
        else:
            out[start:stop] = make_model().fit(Xtr, ytr).predict(Xte)
    return out


# --------------------------------------------------------------------------- metrics
def auc_perm_pvalue(y, score, auc, rng, n_perm):
    """One-sided p-value: how often does a shuffled label reach this AUC by luck?"""
    better = sum(roc_auc_score(rng.permutation(y), score) >= auc for _ in range(n_perm))
    return (1 + better) / (1 + n_perm)


def clf_metrics(y, score, rng, n_perm, with_pvalue=True):
    y = np.asarray(y).astype(int)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, score)
    return {
        "n_oos": len(y), "pos_rate": round(y.mean(), 3),
        "accuracy": round((pred == y).mean(), 3),
        "balanced_acc": round(balanced_accuracy_score(y, pred), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
        "mcc": round(matthews_corrcoef(y, pred), 3),
        "roc_auc": round(auc, 3),
        "pr_auc": round(average_precision_score(y, score), 3),
        "auc_p_value": round(auc_perm_pvalue(y, score, auc, rng, n_perm), 3) if with_pvalue else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def block_bootstrap_delta_auc(y, s_new, s_base, rng, n_boot, block):
    """AUC(new) - AUC(base) with a 95% CI from a moving-block bootstrap over days.
    Blocks keep the day-to-day autocorrelation that an ordinary bootstrap would destroy."""
    y = np.asarray(y).astype(int)
    n = len(y)
    delta = roc_auc_score(y, s_new) - roc_auc_score(y, s_base)
    n_blocks = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_new[idx]) - roc_auc_score(y[idx], s_base[idx]))
    diffs = np.array(diffs)
    return delta, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), float((diffs <= 0).mean())


def reg_metrics(y, pred, base_pred):
    """Out-of-sample R^2 against 'always predict the training mean' (negative = worse than the mean)."""
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - base_pred) ** 2)
    rho, p = spearmanr(y, pred) if np.ptp(pred) > 0 else (np.nan, np.nan)
    return {"n_oos": len(y), "r2_oos": round(r2, 3),
            "spearman_rho": round(rho, 3) if rho == rho else np.nan,
            "spearman_p": round(p, 4) if p == p else np.nan}


def backtest(ret, signal, cost):
    """Long from today's open to the next open when signal=1, otherwise cash.
    Open-to-open returns chain, so staying long costs nothing; each entry/exit costs `cost`."""
    ret, pos = np.asarray(ret, float), np.asarray(signal, float)
    prev = np.r_[0.0, pos[:-1]]
    r = pos * ret - cost * np.abs(pos - prev)
    equity = np.cumprod(1 + r)
    dd = (equity / np.maximum.accumulate(equity) - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return {"total_return": round(equity[-1] - 1, 3), "sharpe": round(sharpe, 2),
            "max_drawdown": round(dd, 3), "time_in_market": round(pos.mean(), 2),
            "n_trades": int(np.abs(pos - prev).sum()),
            "hit_rate": round((ret[pos == 1] > 0).mean(), 2) if pos.sum() else np.nan}, equity


# --------------------------------------------------------------------------- main experiment
def run_stock(stock, models, rng, args):
    df = pd.read_csv(config.PROC_DIR / f"daily_{stock}.csv", parse_dates=["trade_date"])
    n, mt, step = len(df), config.DAILY_MIN_TRAIN, config.DAILY_STEP
    oos = np.arange(mt, n)
    clf_rows, reg_rows, delta_rows, strat_rows, equities = [], [], [], [], {}

    # ---- classification: "significant" move and "direction"
    scores = {}
    for task, ycol in TASKS.items():
        y = df[ycol].values
        for fs, cols in config.DAILY_FEATURE_SETS.items():
            for m in models:
                s = walk_forward(df[cols].values, y, lambda m=m: make_classifier(m, config.SEED),
                                 "clf", mt, step)
                scores[(task, fs, m)] = s[oos]
                met = clf_metrics(y[oos], s[oos], rng, config.N_PERM, with_pvalue=(m != "majority"))
                clf_rows.append({"stock": stock, "task": task, "features": fs, "model": m, **met})

        # does adding WSB to the price features help?  (block-bootstrap CI on the AUC gain)
        for m in models:
            if m == "majority":
                continue
            d, lo, hi, p = block_bootstrap_delta_auc(
                y[oos], scores[(task, "price_plus_wsb", m)], scores[(task, "price_only", m)],
                rng, config.N_BOOT, config.BOOT_BLOCK)
            delta_rows.append({
                "stock": stock, "task": task, "model": m,
                "auc_price_only": round(roc_auc_score(y[oos], scores[(task, "price_only", m)]), 3),
                "auc_price_plus_wsb": round(roc_auc_score(y[oos], scores[(task, "price_plus_wsb", m)]), 3),
                "delta_auc": round(d, 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                "p_no_gain": round(p, 3)})

    # ---- regression: how big will the move be?  (log |move|)
    y = df["log_abs_ret"].values
    base = walk_forward(df[config.DAILY_PRICE_FEATURES].values, y,
                        lambda: make_regressor("mean"), "reg", mt, step)[oos]
    for fs, cols in config.DAILY_FEATURE_SETS.items():
        for m in REGRESSORS:
            if m == "mean":
                continue
            pred = walk_forward(df[cols].values, y, lambda m=m: make_regressor(m, config.SEED),
                                "reg", mt, step)[oos]
            reg_rows.append({"stock": stock, "features": fs, "model": m, **reg_metrics(y[oos], pred, base)})

    # ---- trading: long if P(up) >= 0.5, else cash
    ret = df["ret_fwd"].values[oos]
    bh, bh_eq = backtest(ret, np.ones(len(ret)), config.TRANSACTION_COST)
    equities["buy_and_hold"] = bh_eq
    for fs in config.DAILY_FEATURE_SETS:
        for m in models:
            sig = (scores[("direction", fs, m)] >= 0.5).astype(int)
            res, eq = backtest(ret, sig, config.TRANSACTION_COST)
            strat_rows.append({"stock": stock, "features": fs, "model": m, **res,
                               "buy_hold_return": bh["total_return"], "buy_hold_sharpe": bh["sharpe"],
                               "buy_hold_max_drawdown": bh["max_drawdown"], "test_days": len(ret)})
            if fs == "price_plus_wsb" and m == PRIMARY:
                equities["strategy"] = eq
    equities["dates"] = df["trade_date"].values[oos]

    # ---- permutation importance of each feature (single 70/30 split, "significant" task)
    imp_rows = []
    cols = config.DAILY_FEATURE_SETS["price_plus_wsb"]
    cut = int(n * 0.7)
    ytr, yte = df["significant"].values[:cut], df["significant"].values[cut:]
    if len(np.unique(ytr)) == 2 and len(np.unique(yte)) == 2:
        sc = StandardScaler().fit(df[cols].values[:cut])
        model = make_classifier("rf", config.SEED).fit(sc.transform(df[cols].values[:cut]), ytr)
        pi = permutation_importance(model, sc.transform(df[cols].values[cut:]), yte, scoring="roc_auc",
                                    n_repeats=30, random_state=config.SEED)
        for c, mu, sd in zip(cols, pi.importances_mean, pi.importances_std):
            imp_rows.append({"stock": stock, "feature": c,
                             "group": "price" if c in config.DAILY_PRICE_FEATURES else "wsb",
                             "mean_auc_drop": round(mu, 4), "std": round(sd, 4)})
    return clf_rows, reg_rows, delta_rows, strat_rows, imp_rows, equities


def make_plots(clf, imp, equities):
    colors = {"price_only": "#7f8c8d", "wsb_only": "#e67e22", "price_plus_wsb": "#16a085"}
    stocks = list(clf["stock"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, task in zip(axes, ["significant", "direction"]):
        sub = clf[(clf["task"] == task) & (clf["model"] == PRIMARY)]
        w = 0.26
        for i, fs in enumerate(config.DAILY_FEATURE_SETS):
            vals = [sub[(sub["stock"] == s) & (sub["features"] == fs)]["roc_auc"].iloc[0] for s in stocks]
            ax.bar(np.arange(len(stocks)) + (i - 1) * w, vals, w, label=fs, color=colors[fs])
        ax.axhline(0.5, color="k", lw=1, ls="--")
        ax.set_xticks(range(len(stocks))); ax.set_xticklabels(stocks)
        ax.set_title(f"Task: {task} (walk-forward, logistic regression)")
        ax.set_ylim(0.3, 1.0); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("out-of-sample ROC-AUC (0.5 = coin flip)")
    axes[0].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "daily_auc_by_featureset.png", dpi=150); plt.close()

    if len(imp):
        fig, axes = plt.subplots(1, len(stocks), figsize=(4 * len(stocks), 3.8))
        for ax, s in zip(np.atleast_1d(axes), stocks):
            sub = imp[imp["stock"] == s].sort_values("mean_auc_drop")
            if sub.empty:
                ax.set_title(f"{s} (skipped)"); continue
            ax.barh(sub["feature"], sub["mean_auc_drop"], xerr=sub["std"],
                    color=["#e67e22" if g == "wsb" else "#7f8c8d" for g in sub["group"]])
            ax.set_title(s); ax.grid(alpha=.3, axis="x")
            ax.set_xlabel("AUC drop when shuffled")
        plt.suptitle("Permutation importance, 'significant move' (orange = WSB, grey = price)")
        plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "daily_importance.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, len(equities), figsize=(4 * len(equities), 3.6))
    for ax, (s, eq) in zip(np.atleast_1d(axes), equities.items()):
        d = eq["dates"]
        ax.plot(d, eq["buy_and_hold"], label="buy & hold", color="#7f8c8d")
        ax.plot(d, eq["strategy"], label="model (logistic, price+WSB)", color="#16a085")
        ax.axhline(1, color="k", lw=.6); ax.set_title(s); ax.grid(alpha=.3)
        ax.tick_params(axis="x", rotation=45)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    plt.suptitle(f"Growth of 1 unit, out-of-sample days, {config.TRANSACTION_COST * 1e4:.0f} bps cost per trade")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "daily_strategy_equity.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the slow models (random forest, MLP)")
    args = ap.parse_args()
    models = [m for m in CLASSIFIERS if not (args.fast and m in ("rf", "mlp"))]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    clf, reg, delta, strat, imp, equities = [], [], [], [], [], {}
    for stock in config.MODEL_STOCKS:
        if not (config.PROC_DIR / f"daily_{stock}.csv").exists():
            print(f"{stock}: run `python -m src.build_daily` first"); continue
        print(f"=== {stock} ===", flush=True)
        a, b, c, d, e, eq = run_stock(stock, models, rng, args)
        clf += a; reg += b; delta += c; strat += d; imp += e; equities[stock] = eq

    clf, reg, delta = pd.DataFrame(clf), pd.DataFrame(reg), pd.DataFrame(delta)
    strat, imp = pd.DataFrame(strat), pd.DataFrame(imp)
    clf.to_csv(config.RESULTS_DIR / "daily_classification.csv", index=False)
    reg.to_csv(config.RESULTS_DIR / "daily_regression.csv", index=False)
    delta.to_csv(config.RESULTS_DIR / "daily_incremental_value.csv", index=False)
    strat.to_csv(config.RESULTS_DIR / "daily_strategy.csv", index=False)
    imp.to_csv(config.RESULTS_DIR / "daily_importance.csv", index=False)
    make_plots(clf, imp, equities)

    show = ["stock", "features", "model", "accuracy", "balanced_acc", "f1", "roc_auc", "auc_p_value"]
    for task in TASKS:
        print(f"\n=== {task}: logistic regression, out-of-sample ===")
        print(clf[(clf["task"] == task) & (clf["model"] == PRIMARY)][show].to_string(index=False))
    print("\n=== Does adding WSB to the price features help? (AUC gain, 95% block-bootstrap CI) ===")
    print(delta[delta["model"] == PRIMARY].to_string(index=False))
    print("\n=== Regression of log|move| (R2 vs training-mean baseline) ===")
    print(reg.to_string(index=False))
    print("\n=== Trading, price+WSB features ===")
    print(strat[strat["features"] == "price_plus_wsb"].drop(columns="features").to_string(index=False))
    print(f"\nSaved tables and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
