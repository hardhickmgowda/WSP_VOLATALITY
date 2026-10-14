"""Live demo: type a WallStreetBets-style post and get a prediction.

Run (after training):
  python -m src.predict --stock GME --text "GME to the moon, holding forever" --score 500 --comments 120
  python -m src.predict --stock TSLA            (asks you to type the post)

Uses the model saved by `python -m src.train` (default: direction label, random split).
"""
import argparse

import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src import config
from src.preprocess import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", choices=config.MODEL_STOCKS, default="GME")
    ap.add_argument("--text", default=None, help="the post (title + body)")
    ap.add_argument("--score", type=int, default=100, help="upvotes of the post")
    ap.add_argument("--comments", type=int, default=20, help="number of comments")
    ap.add_argument("--label", choices=["direction", "significant"], default="direction")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    args = ap.parse_args()

    path = config.MODELS_DIR / f"{args.stock}_{args.label}_{args.split}.pkl"
    if not path.exists():
        raise SystemExit(f"No saved model at {path}. Run: python -m src.train "
                         f"--label {args.label} --split {args.split}")
    bundle = joblib.load(path)

    text = args.text or input("Type a post: ")
    s = SentimentIntensityAnalyzer().polarity_scores(text.lower())
    feats = pd.DataFrame([{
        "post_score": args.score,
        "total_comms_num": args.comments,
        "len_of_post": len(clean_text(text).split()),
        "positive_sentiment": s["pos"],
        "neutral_sentiment": s["neu"],
        "negative_sentiment": s["neg"],
    }])[config.FEATURES]
    X = bundle["scaler"].transform(feats)

    p_lr = float(bundle["logreg"].predict_proba(X)[0])
    p_nn = float(bundle["nn"].predict_proba(X)[0, 1])
    what = "next-day open HIGHER" if args.label == "direction" else "a SIGNIFICANT (bigger than usual) move"

    print("\nFeatures:", feats.iloc[0].round(3).to_dict())
    print(f"\nStock: {args.stock}   question: will there be {what}?")
    print(f"  Logistic regression : P = {p_lr:.2f}  -> {'YES' if p_lr >= .5 else 'NO'}")
    print(f"  Neural network      : P = {p_nn:.2f}  -> {'YES' if p_nn >= .5 else 'NO'}")


if __name__ == "__main__":
    main()
