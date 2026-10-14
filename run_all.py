"""Run the whole project from start to end with one command.

  python run_all.py              # everything
  python run_all.py --skip-prices   # reuse already-downloaded price files (offline demo)
  python run_all.py --fast          # shorter run: skips the slowest daily models

The dataset is downloaded automatically (needs internet the first time).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(module, *extra):
    cmd = [sys.executable, "-m", module, *extra]
    print("\n" + "=" * 70 + f"\n>>> {' '.join(cmd[1:])}\n" + "=" * 70, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--fast", action="store_true", help="skip the slowest daily models (random forest, MLP)")
    args = ap.parse_args()

    run("src.download_data")          # downloads the posts from OpenML if not already present
    run("src.preprocess")
    run("src.eda")
    if not args.skip_prices:
        run("src.fetch_prices")
    run("src.build_dataset")
    run("src.train", "--label", "direction", "--split", "random")    # paper's setup
    run("src.train", "--label", "direction", "--split", "time")      # leakage-free setup
    run("src.train", "--label", "significant", "--split", "time")    # volatility label
    run("src.volatility_analysis")                                   # posts vs size of moves
    run("src.build_daily")                                           # causal daily panel (no look-ahead)
    run("src.walk_forward", *(["--fast"] if args.fast else []))      # walk-forward models, ablation, strategy
    run("src.daily_correlation")                                     # attention vs move size, controlling for past volatility
    print("\nDone. All tables and plots are in results/, trained models in models/.")
    print('Try the demos:')
    print('  python -m src.predict --stock GME --text "GME to the moon" --score 500 --comments 100')
    print("  python -m src.predict_daily --stock GME --date 2021-05-12")


if __name__ == "__main__":
    main()
