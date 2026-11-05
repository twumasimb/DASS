from random import seed, sample


class RANDOM:
    """
    Source-only baseline: randomly selects a training subset from a source dataset.

    Expected:
    - Source dataset: a HuggingFace Dataset (not a DatasetDict)
    """

    def __init__(self, source, budget, random_seed=42):
        self.source = source
        self.budget = budget
        self.random_seed = random_seed

        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(f"Budget must be a positive integer, got {budget}")

        required_columns = ["input", "label"] if "input" in source.column_names else ["image", "label"]
        for column in required_columns:
            if column not in source.column_names:
                raise ValueError(f"Source dataset missing required column: {column}")

    def curate(self):
        seed(self.random_seed)
        selected_indices = sample(range(len(self.source)), self.budget)
        selected_source = self.source.select(selected_indices).add_column("domain", ["source"] * self.budget)

        return selected_source
