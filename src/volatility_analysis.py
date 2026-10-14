"""Step 5: does WSB activity relate to VOLATILITY (size of the move) or DIRECTION?

For each stock and trading day we count posts and compare with
  |move| = |Open(next day) / Open(today) - 1|   (volatility)
  move   =  Open(next day) / Open(today) - 1    (direction)
using Spearman rank correlation (robust to the huge outlier days of Jan 2021).

"same day" : posts assigned to trading day d vs the move from d's open to d+1's open
              (some posts are written after the open, so this is an association)
"lagged"   : posts of the PREVIOUS trading day vs today's move (a true forecast)

Run:  python -m src.volatility_analysis
Outputs: results/volatility_correlation.csv, results/posts_vs_volatility.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from src import config


def main():
    rows = []
    fig, axes = plt.subplots(1, len(config.MODEL_STOCKS), figsize=(4 * len(config.MODEL_STOCKS), 3.6))
    for ax, stock in zip(axes, config.MODEL_STOCKS):
        d = pd.read_csv(config.PROC_DIR / f"dataset_{stock}.csv", parse_dates=["trade_date"])
        day = d.groupby("trade_date").agg(posts=("id", "size"),
                                          move=("ret_next_open", "first"),
                                          pos_sent=("positive_sentiment", "mean"),
                                          neg_sent=("negative_sentiment", "mean"))
        day["abs_move"] = day["move"].abs()
        day["posts_prev"] = day["posts"].shift(1)
        lag = day.dropna(subset=["posts_prev"])

        def rho(x, y):
            r, p = spearmanr(x, y)
            return round(r, 3), round(p, 4)

        for name, x, y, data in [
            ("posts vs |move| (same day)", "posts", "abs_move", day),
            ("posts vs |move| (lagged)", "posts_prev", "abs_move", lag),
            ("posts vs direction (same day)", "posts", "move", day),
            ("positive sentiment vs direction", "pos_sent", "move", day),
        ]:
            r, p = rho(data[x], data[y])
            rows.append({"stock": stock, "test": name, "days": len(data), "spearman_rho": r,
                         "p_value": p, "significant_5pct": p < 0.05})

        ax.scatter(day["posts"], day["abs_move"] * 100, s=12, alpha=.6, color="teal")
        ax.set_xscale("log")
        r, p = rho(day["posts"], day["abs_move"])
        ax.set_title(f"{stock}  (ρ={r:.2f}, p={p:.3f})")
        ax.set_xlabel("WSB posts that day (log)")
        ax.set_ylabel("|next-open move| %")
        ax.grid(alpha=.3)

    plt.suptitle("Does WSB activity go with bigger price moves?")
    plt.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(config.RESULTS_DIR / "posts_vs_volatility.png", dpi=150)

    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "volatility_correlation.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
