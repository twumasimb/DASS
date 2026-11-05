"""Run DASS and its baselines end to end: select data, fine-tune RoBERTa, evaluate.

Example:
    python main.py --experiment_name Experimental_Results --dataset amazon-yelp --source_method fl1mi
"""

import argparse
import os
import shutil

from datasets import Dataset, load_from_disk

from dass import DASS, FL_RANDOM, RANDOM, RANDOM_DASS
from dass.sentiment_classifier import SentimentClassifier
from dass.utils import split_source_target

# method name -> (selector class, output folder)
METHODS = {
    "RANDOM_DASS": (RANDOM_DASS, "rand_dass_output"),
    "RANDOM": (RANDOM, "random_output"),
    "DASS": (DASS, "dass_output"),
    "FL_RANDOM": (FL_RANDOM, "fl_rand_output"),
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _to_input_label(dataset, text_col="text", label_col="label"):
    """Keep only the text/label columns and rename them to input/label."""
    return Dataset.from_dict({"input": dataset[text_col], "label": dataset[label_col]})


def load_amazon_yelp(data_dir, num_source=10000):
    """Amazon reviews (source) -> Yelp restaurant reviews (target)."""
    amazon = load_from_disk(os.path.join(data_dir, "amazon-yelp-dataset", "english-amazon-review-sentiment"))
    yelp = load_from_disk(os.path.join(data_dir, "amazon-yelp-dataset", "yelp-reviews-split"))

    source = _to_input_label(amazon["train"].select(range(num_source)))
    target = _to_input_label(yelp["train"])
    val = _to_input_label(yelp["validation"])
    test = _to_input_label(yelp["test"])
    return source, target, val, test


def load_amazon_subgroup(data_dir, target_domain, balanced=False):
    """Leave-one-category-out over Amazon categories (Appliances/Beauty/Fashion/Magazine).

    The target category is split 50/25/25 into selection pool / val / test.
    """
    name = "final_dataset_balanced" if balanced else "final_dataset"
    ds = load_from_disk(os.path.join(data_dir, name))
    source, target = split_source_target(ds, target_domain)

    n = len(target)
    half, three_quarters = n // 2, n // 2 + (n - n // 2) // 2
    target_pool = target.select(range(0, half))
    val = target.select(range(half, three_quarters))
    test = target.select(range(three_quarters, n))
    val = val.add_column("domain", ["target"] * len(val))
    test = test.add_column("domain", ["target"] * len(test))
    return source, target_pool, val, test


# --------------------------------------------------------------------------- #
# Experiment loop
# --------------------------------------------------------------------------- #
def run_experiment(args, source_dataset, target_dataset, val_dataset, test_dataset):
    if args.wandb:
        import wandb

    for method in args.methods:
        selector_cls, output_prefix = METHODS[method]
        run_dir = os.path.join(args.experiment_name, args.source_method, output_prefix)

        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.experiment_name}_{method}",
                config={"method": method, **vars(args)},
            )

        selector_kwargs = {"source": source_dataset, "budget": args.budget}
        if method != "RANDOM":
            selector_kwargs.update(
                {
                    "target": target_dataset,
                    "model": args.model,
                    "target_ratio": args.target_ratio,
                    "source_method": args.source_method,
                    "source_eta": args.source_eta,
                }
            )
        selector = selector_cls(**selector_kwargs)
        curated_train_dataset = selector.curate()

        if args.save_stats and method != "RANDOM":
            selector.generate_statistics(output_dir=os.path.join(run_dir, "stats"))

        classifier = SentimentClassifier()
        classifier.prepare_data(train_data=curated_train_dataset, dev_data=val_dataset)

        model_training_dir = os.path.join(run_dir, "model_training")
        classifier.train(
            output_dir=model_training_dir,
            experiment_name=args.experiment_name,
            epochs=args.epochs,
            lr=args.learning_rate,
            report_to="wandb" if args.wandb else "none",
        )

        results_dir = os.path.join(run_dir, "results")
        results = classifier.evaluate(test_data=test_dataset, output_dir=results_dir)
        classifier.visualize(output_dir=results_dir)

        if args.wandb:
            wandb.log({k: v for k, v in results.items() if not isinstance(v, dict)})
            wandb.finish()

        # Model checkpoints are large; only metrics and plots are kept.
        if not args.keep_checkpoints and os.path.exists(model_training_dir):
            print(f"Cleaning up model training directory: {model_training_dir}")
            shutil.rmtree(model_training_dir)


def get_args():
    parser = argparse.ArgumentParser(description="Run DASS and baseline data-selection experiments")

    parser.add_argument("--experiment_name", type=str, required=True, help="Output folder for this experiment")
    parser.add_argument("--dataset", choices=["amazon-yelp", "amazon-subgroup"], default="amazon-yelp")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Folder holding the processed datasets")
    parser.add_argument("--target_domain", choices=["Appliances", "Beauty", "Fashion", "Magazine"],
                        default="Appliances", help="Target category for --dataset amazon-subgroup")
    parser.add_argument("--balanced", action="store_true", help="Use the label-balanced Amazon subgroup dataset")

    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS),
                        help="Which selection methods to run")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence embedding model")
    parser.add_argument("--budget", type=int, default=1000, help="Total number of selected training examples")
    parser.add_argument("--target_ratio", type=float, default=0.3, help="Fraction of the budget taken from the target domain")
    parser.add_argument("--source_method", choices=["fl1mi", "fl2mi", "gcmi", "logdetmi"], default="fl1mi",
                        help="Submodular mutual information function for source selection")
    parser.add_argument("--source_eta", type=float, default=0.8, help="SMI trade-off parameter (eta)")

    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for fine-tuning")

    parser.add_argument("--save_stats", action="store_true", help="Save selection statistics and t-SNE/PCA plots")
    parser.add_argument("--keep_checkpoints", action="store_true", help="Keep model checkpoints after evaluation")
    parser.add_argument("--wandb", action="store_true", help="Log runs to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="Domain-Adaptation")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.dataset == "amazon-yelp":
        source, target, val, test = load_amazon_yelp(args.data_dir)
    else:
        source, target, val, test = load_amazon_subgroup(args.data_dir, args.target_domain, args.balanced)

    print(f"source={len(source)}  target pool={len(target)}  val={len(val)}  test={len(test)}")
    run_experiment(args, source, target, val, test)
