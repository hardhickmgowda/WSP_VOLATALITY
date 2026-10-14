"""Step 4: train + evaluate the models for every stock, and backtest a simple strategy.

Run examples:
  python -m src.train                          # paper setup: direction label, random split
  python -m src.train --split time             # honest setup: train on past, test on future
  python -m src.train --label significant      # "big move" label (volatility)

Always compare model accuracy with majority_acc (always guessing the most
common class): beating 50% means nothing if 77% of the labels are "down".

Outputs (in results/):
  results_<label>_<split>.csv     accuracy table (like the paper's Table 1)
  convergence_<label>_<split>.png Newton's-method convergence (like Figure 3)
  strategy_<label>_<split>.csv    trading strategy vs buy-and-hold
"""
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 30 epochs is the paper's choice

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import NewtonLogisticRegression, RandomBaseline, build_nn


def split_data(df, how):
    """80 / 10 / 10 train / validation / test."""
    if how == "random":     # what the paper did
        train, rest = train_test_split(df, test_size=0.2, random_state=config.SEED)
        val, test = train_test_split(rest, test_size=0.5, random_state=config.SEED)
    else:                   # chronological: no peeking at the future
        # cut by share of POSTS (not days), never splitting one day across sets,
        # so the test set is ~10% of posts even though posts cluster in Jan-Feb 2021
        share = df.groupby("trade_date").size().sort_index().cumsum() / len(df)
        d80 = share.index[share.values >= 0.8][0]
        d90 = share.index[share.values >= 0.9][0]
        if d90 <= d80:                      # tiny datasets: make sure all three sets exist
            later = share.index[share.index > d80]
            d90 = later[len(later) // 2] if len(later) > 1 else share.index[-1]
        train = df[df["trade_date"] < d80]
        val = df[(df["trade_date"] >= d80) & (df["trade_date"] < d90)]
        test = df[df["trade_date"] >= d90]
    return train, val, test


def backtest(test_df, preds, label):
    """Each test day: majority vote of all post-level predictions for that day.
    direction label: vote 'up' -> buy at today's open, sell at next open; else stay in cash.
    Compared with buying every day (always long)."""
    if label != "direction":
        return None
    t = test_df.assign(pred=preds)
    daily = t.groupby("trade_date").agg(vote=("pred", "mean"), ret=("ret_next_open", "first"))
    daily["signal"] = (daily["vote"] > 0.5).astype(int)
    strat = (1 + daily["signal"] * daily["ret"]).prod() - 1
    hold = (1 + daily["ret"]).prod() - 1
    day_acc = ((daily["ret"] > 0).astype(int) == daily["signal"]).mean()
    return {"test_days": len(daily), "daily_vote_accuracy": round(day_acc, 3),
            "strategy_return": round(strat, 3), "always_long_return": round(hold, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["direction", "significant"], default="direction")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    args = ap.parse_args()
    ycol = f"label_{args.label}"
    config.RESULTS_DIR.mkdir(exist_ok=True)

    rows, strat_rows = [], []
    plt.figure(figsize=(6, 4))
    for stock in config.MODEL_STOCKS:
        path = config.PROC_DIR / f"dataset_{stock}.csv"
        if not path.exists():
            print(f"{stock}: dataset missing, skipping"); continue
        df = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=config.FEATURES)
        if len(df) < 50:
            print(f"{stock}: only {len(df)} posts, skipping"); continue

        train, val, test = split_data(df, args.split)
        scaler = StandardScaler().fit(train[config.FEATURES])          # fit on train only!
        Xtr, Xva, Xte = (scaler.transform(d[config.FEATURES]) for d in (train, val, test))
        ytr, yva, yte = train[ycol].values, val[ycol].values, test[ycol].values

        # 1) random baseline
        base_pred = RandomBaseline(config.SEED).predict(Xte)
        # 2) logistic regression with Newton's method
        lr = NewtonLogisticRegression().fit(Xtr, ytr)
        lr_pred = lr.predict(Xte)
        plt.semilogy(range(1, len(lr.delta_norms) + 1), lr.delta_norms, marker="o", label=stock)
        # 3) neural network (bigger batch for the big stocks, as in the paper)
        bs = 64 if stock in ("GME", "TSLA") else 16
        nn = build_nn(batch_size=bs).fit(Xtr, ytr)
        nn_pred = nn.predict(Xte)

        rows.append({
            "stock": stock, "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "test_pos_rate": round(yte.mean(), 3),
            "test_days": test["trade_date"].nunique(),
            # always guess the most common class of the TRAINING set
            "majority_acc": round(accuracy_score(yte, np.full(len(yte), int(ytr.mean() >= 0.5))), 3),
            "random_acc": round(accuracy_score(yte, base_pred), 3),
            "logreg_acc": round(accuracy_score(yte, lr_pred), 3) if lr.converged else "not converged",
            "logreg_f1": round(f1_score(yte, lr_pred, zero_division=0), 3),
            "nn_val_acc": round(accuracy_score(yva, nn.predict(Xva)), 3),
            "nn_acc": round(accuracy_score(yte, nn_pred), 3),
            "nn_f1": round(f1_score(yte, nn_pred, zero_division=0), 3),
        })
        # save trained models so src/predict.py can use them in the live demo
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"scaler": scaler, "logreg": lr, "nn": nn, "label": args.label},
                    config.MODELS_DIR / f"{stock}_{args.label}_{args.split}.pkl")

        bt = backtest(test, nn_pred, args.label)
        if bt:
            strat_rows.append({"stock": stock, **bt})

    tag = f"{args.label}_{args.split}"
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / f"results_{tag}.csv", index=False)
    print("\n=== Test results ===\n", res.to_string(index=False))

    plt.xlabel("Iteration"); plt.ylabel("||Δθ|| (log scale)")
    plt.title("Newton's method convergence"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / f"convergence_{tag}.png", dpi=150)

    if strat_rows:
        st = pd.DataFrame(strat_rows)
        st.to_csv(config.RESULTS_DIR / f"strategy_{tag}.csv", index=False)
        print("\n=== Trading strategy on test days (NN majority vote) ===\n", st.to_string(index=False))
    print(f"\nSaved outputs to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
