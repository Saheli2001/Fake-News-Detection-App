import argparse
from src.utils import split_data
from src.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create train-test split
    split_parser = subparsers.add_parser("split", help="Make train-test split")
    split_parser.add_argument("--data_path", type=str, default="data/train.csv")
    split_parser.add_argument("--test_size", type=float, default=0.2)
    split_parser.add_argument("--random_state", type=int, default=42)
    split_parser.add_argument("--output_dir", type=str, default="data")

    # train model
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--feature", type=str, choices=["tfidf", "ngram", "word2vec"], required=True)
    train_parser.add_argument("--model", type=str, choices=["logistic", "random_forest", "xgboost"], required=True)
    train_parser.add_argument("--train_path", type=str, default="data/train.csv")
    train_parser.add_argument("--val_path", type=str, default="data/test.csv")

    args = parser.parse_args()

    if args.command == "split":
        split_data(args.data_path, args.test_size, args.random_state, args.output_dir)

    elif args.command == "train":
        train(args.feature, args.model, args.train_path, args.val_path)


if __name__ == "__main__":
    main()
