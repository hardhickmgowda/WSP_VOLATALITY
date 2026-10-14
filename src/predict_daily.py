"""Live demo for the daily model: pick a trading day and see what the model would have said.

  python -m src.predict_daily --stock GME --date 2021-05-12
  python -m src.predict_daily --stock AMC --date 2021-06-02 --features price_only

The models are trained ONLY on days before the chosen date (exactly like the walk-forward
evaluation), so the demo can't cheat. It then prints the features known before that day's open,
the predicted probabilities, and what really happened from that open to the next open.
"""
import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import make_classifier, positive_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", choices=config.MODEL_STOCKS, default="GME")
    ap.add_argument("--date", required=True, help="trading day to predict, e.g. 2021-05-12")
    ap.add_argument("--features", choices=list(config.DAILY_FEATURE_SETS), default="price_plus_wsb")
    args = ap.parse_args()

    path = config.PROC_DIR / f"daily_{args.stock}.csv"
    if not path.exists():
        raise SystemExit("Run first:  python -m src.build_daily")
    df = pd.read_csv(path, parse_dates=["trade_date"])

    day = pd.Timestamp(args.date)
    hit = df.index[df["trade_date"] == day]
    if len(hit) == 0:
        valid = df["trade_date"].iloc[config.DAILY_MIN_TRAIN:]
        raise SystemExit(f"{args.date} is not in the daily panel. Valid days: {valid.min().date()} "
                         f"to {valid.max().date()} (trading days with enough scraped posts).")
    i = int(hit[0])
    if i < config.DAILY_MIN_TRAIN:
        first = df["trade_date"].iloc[config.DAILY_MIN_TRAIN].date()
        raise SystemExit(f"Need at least {config.DAILY_MIN_TRAIN} earlier days to train; pick {first} or later.")

    cols = config.DAILY_FEATURE_SETS[args.features]
    scaler = StandardScaler().fit(df[cols].iloc[:i])
    Xtr, Xte = scaler.transform(df[cols].iloc[:i]), scaler.transform(df[cols].iloc[[i]])
    row = df.iloc[i]

    print(f"\n{args.stock} on {day.date()}  (trained on {i} earlier days, features: {args.features})")
    print(f"Posts about {args.stock} before the open: {int(row['n_posts'])}")
    print("Features:", {c: round(float(row[c]), 3) for c in cols})
    print(f"\nQuestion 1: will the next open-to-open move be BIGGER than usual (> {row['sig_thr']:.1%})?")
    print(f"Question 2: will the next open be HIGHER than today's open?\n")
    for task, name in [("significant", "bigger-than-usual move"), ("direction", "price goes up")]:
        for m in ["majority", "logreg", "hgb"]:
            model = make_classifier(m, config.SEED).fit(Xtr, df[task].iloc[:i].astype(int))
            p = float(positive_proba(model, Xte)[0])
            print(f"  {name:<24s} {m:<9s} P = {p:.2f} -> {'YES' if p >= 0.5 else 'NO'}")
        print()
    print(f"What actually happened: move = {row['ret_fwd']:+.1%}  ->  "
          f"{'BIGGER than usual' if row['significant'] else 'normal-sized'}, "
          f"{'UP' if row['direction'] else 'DOWN'}")


if __name__ == "__main__":
    main()
