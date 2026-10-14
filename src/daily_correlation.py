"""Step 7: simple, low-variance check on the causal daily panel.

Models with 15 features and ~100 rows overfit easily, so here we ask the question directly:
"Is WSB attention related to the SIZE of the next open-to-open move, once we control for how
volatile the stock already was?"  We use rank (Spearman) correlations, because squeeze days are
extreme outliers, and PARTIAL rank correlations that remove the effect of past volatility
(log_vol5, log_vol20, log_range1) from both variables first.

Rows: all days of the panel, and only days AFTER the first DAILY_MIN_TRAIN days (so the unusual
Feb-2021 squeeze period does not drive the result).
P-values assume independent days; volatility is autocorrelated, so treat them as optimistic.

Run:  python -m src.daily_correlation
Output: results/daily_correlation.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as t_dist

from src import config

CONTROLS = ["log_vol5", "log_vol20", "log_range1"]


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after removing the (rank) effect of the control variables."""
    rx, ry = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(x))])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    dof = len(x) - 2 - len(controls)
    tstat = r * np.sqrt(dof / (1 - r ** 2))
    return r, float(2 * t_dist.sf(abs(tstat), dof))


def main():
    rows = []
    for stock in config.MODEL_STOCKS:
        path = config.PROC_DIR / f"daily_{stock}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["trade_date"])
        df["net_sent"] = df["mean_pos"] - df["mean_neg"]
        for period, d in [("all days", df), ("after first 40 days", df.iloc[config.DAILY_MIN_TRAIN:])]:
            for name, x, y in [("log posts vs size of next move", "log_posts", "log_abs_ret"),
                               ("attention shock vs size of next move", "attn_shock", "log_abs_ret"),
                               ("sentiment vs direction of next move", "net_sent", "ret_fwd")]:
                r, p = spearmanr(d[x], d[y])
                rp, pp = partial_spearman(d[x].values, d[y].values, [d[c].values for c in CONTROLS])
                rows.append({"stock": stock, "period": period, "relationship": name, "days": len(d),
                             "rho": round(r, 3), "p": round(p, 4),
                             "partial_rho": round(rp, 3), "partial_p": round(pp, 4)})
    out = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "daily_correlation.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved to {config.RESULTS_DIR / 'daily_correlation.csv'}")


if __name__ == "__main__":
    main()
