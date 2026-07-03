"""
load_model.py
--------------
Loads the pre-trained model bundle (trained_models.pkl) and generates
recommendations, without retraining anything. This is the fast path for
using the model after train_and_save.py has been run once.

Usage:
    python load_model.py --user 7 --n 10
"""

import argparse
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "trained_models.pkl"


def main():
    parser = argparse.ArgumentParser(description="Load trained recommender and get recommendations")
    parser.add_argument("--user", type=int, required=True, help="user_id to recommend for")
    parser.add_argument("--n", type=int, default=5, help="number of recommendations")
    parser.add_argument("--method", choices=["content", "collaborative", "hybrid"],
                         default="hybrid", help="which model to use")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python train_and_save.py` first."
        )

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    if args.method == "content":
        recs = bundle["content"].recommend(args.user, n=args.n)
    elif args.method == "collaborative":
        recs = bundle["collaborative"].recommend_svd(args.user, n=args.n)
    else:
        recs = bundle["hybrid"].recommend(args.user, n=args.n)

    print(f"Top {args.n} '{args.method}' recommendations for user {args.user}:\n")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
