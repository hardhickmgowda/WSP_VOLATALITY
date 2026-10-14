"""Step 0: download the WallStreetBets posts dataset (no account needed).

The original Kaggle page (gpreda/reddit-wallstreetbets-posts) has been taken down.
The same dataset (53,187 posts, same 8 columns, CC0 licence) is mirrored on
OpenML: https://www.openml.org/d/43373

Run:  python -m src.download_data
Output: data/raw/reddit_wsb.csv
"""
import pandas as pd
from sklearn.datasets import fetch_openml

from src import config

OPENML_ID = 43373
COLUMNS = ["title", "score", "id", "url", "comms_num", "created", "body", "timestamp"]


def main():
    if config.POSTS_CSV.exists():
        print(f"{config.POSTS_CSV} already exists, nothing to do.")
        return
    print(f"Downloading OpenML dataset {OPENML_ID} (Reddit-WallStreetBets-Posts)...")
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    df = bunch.frame.copy()

    # OpenML marks 'id' as an ignored attribute, so it may be dropped: rebuild it
    if "id" not in df.columns:
        df["id"] = [f"post_{i}" for i in range(len(df))]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]
    # categorical/object -> plain strings, keep missing values as NaN
    for col in ["title", "url", "body", "timestamp"]:
        df[col] = df[col].astype("object").where(df[col].notna(), None)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.POSTS_CSV, index=False)
    print(f"Saved {len(df)} posts -> {config.POSTS_CSV}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
