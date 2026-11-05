"""Domain-Aware Submodular Selection (DASS) and its ablation variants.

All selectors share the same two-stage recipe and differ only in how each stage
picks its indices:

    =============  ==========================  ==============================
    Selector       Stage 1: target (annotate)  Stage 2: source (transfer)
    =============  ==========================  ==============================
    DASS           Facility Location           Submodular Mutual Information
    RANDOM_DASS    Random                      Submodular Mutual Information
    FL_RANDOM      Facility Location           Random
    =============  ==========================  ==============================
"""

from random import sample, seed

from datasets import concatenate_datasets

from . import plot_utils
from .smi import SMI, SubmodSelection
from .statistics_generator import StatisticsGenerator
from .utils import compute_image_embeddings, compute_text_embeddings


class DASS:
    """
    Domain-Aware Submodular Selection.

    Selects ``budget`` training examples: ``budget * target_ratio`` from the
    target domain (to be annotated) and the rest from the labeled source domain.

    Datasets must be HuggingFace ``Dataset`` objects with columns:
      - text data:  ``input`` and ``label``
      - image data: ``image`` and ``label``
    """

    def __init__(
        self,
        source,
        target,
        model,
        budget,
        target_ratio=None,
        datatype: str = "text",
        random_seed: int = 42,
        # Stage 1 (target) parameters
        target_selection_method: str = "fl",
        target_optimizer: str = "LazyGreedy",
        # Stage 2 (source) parameters
        source_optimizer: str = "LazyGreedy",
        source_method: str = "fl1mi",
        source_metric: str = "cosine",
        source_eta: float = 1.0,
        source_nu: float = 1.0,
        source_lambda: float = 1.0,
        source_verbose: bool = False,
        source_stop_if_zero_gain: bool = False,
        source_stop_if_negative_gain: bool = False,
    ):
        self.source = source
        self.target = target
        self.model = model
        self.random_seed = random_seed
        self.budget = budget
        self.datatype = datatype
        self.target_ratio = target_ratio

        self.target_selection_method = target_selection_method
        self.target_optimizer = target_optimizer

        self.source_optimizer = source_optimizer
        self.source_method = source_method
        self.source_metric = source_metric
        self.source_eta = source_eta
        self.source_nu = source_nu
        self.source_lambda = source_lambda
        self.source_verbose = source_verbose
        self.source_stop_if_zero_gain = source_stop_if_zero_gain
        self.source_stop_if_negative_gain = source_stop_if_negative_gain

        self._last_selected_source_indices = None
        self._last_selected_target_indices = None
        self._cached_embeddings = None

        valid_datatypes = ["text", "image"]
        if self.datatype not in valid_datatypes:
            raise ValueError(f"Invalid datatype: {self.datatype}. Expected one of: {valid_datatypes}")

        if not isinstance(self.budget, int) or self.budget <= 0:
            raise ValueError(f"Budget must be a positive integer, got {self.budget}")

        if self.target_ratio is not None and not 0 <= self.target_ratio <= 1:
            raise ValueError(f"Target ratio must be between 0 and 1, got {self.target_ratio}")

        required_columns = ["input", "label"] if self.datatype == "text" else ["image", "label"]
        for column in required_columns:
            if column not in self.source.column_names:
                raise ValueError(f"Source dataset missing required column: {column}")
            if column not in self.target.column_names:
                raise ValueError(f"Target dataset missing required column: {column}")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def compute_embeddings(self):
        if self.datatype == "text":
            source_embeddings = compute_text_embeddings(self.source["input"], model=self.model)
            target_embeddings = compute_text_embeddings(self.target["input"], model=self.model)
        else:
            source_embeddings = compute_image_embeddings(self.source["image"], model=self.model)
            target_embeddings = compute_image_embeddings(self.target["image"], model=self.model)
        return source_embeddings, target_embeddings

    # ------------------------------------------------------------------ #
    # Selection stages (overridden by the ablation variants)
    # ------------------------------------------------------------------ #
    def select_target(self, target_embeddings, target_budget):
        """Stage 1: pick representative target points with a submodular function."""
        selector = SubmodSelection(
            target_domain_embeddings=target_embeddings,
            method=self.target_selection_method,
            optimizer=self.target_optimizer,
        )
        return selector.select(budget=target_budget)

    def select_source(self, source_embeddings, selected_target_embeddings, source_budget):
        """Stage 2: pick source points that maximize SMI with the selected target set."""
        smi = SMI(
            target_domain_embeddings=selected_target_embeddings,
            source_domain_embeddings=source_embeddings,
            optimizer=self.source_optimizer,
            method=self.source_method,
            metric=self.source_metric,
            eta=self.source_eta,
            nu=self.source_nu,
            lambdaVal=self.source_lambda,
            verbose=self.source_verbose,
            stopIfZeroGain=self.source_stop_if_zero_gain,
            stopIfNegativeGain=self.source_stop_if_negative_gain,
        )
        return smi.select(budget=source_budget)

    def _random_indices(self, population_size, k):
        seed(self.random_seed)
        return sample(range(population_size), k)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def _split_budget(self):
        if self.target_ratio is None:
            target_budget = len(self.target)
        else:
            target_budget = int(self.budget * self.target_ratio)
        source_budget = self.budget - target_budget

        if target_budget <= 0 or source_budget <= 0:
            raise ValueError(
                f"Invalid budget allocation: target={target_budget}, source={source_budget}. "
                "Check budget and target_ratio."
            )
        return target_budget, source_budget

    def curate(self):
        """Run both selection stages and return the combined training ``Dataset``.

        The returned dataset has an extra ``domain`` column (``"source"``/``"target"``).
        """
        target_budget, source_budget = self._split_budget()
        source_embeddings, target_embeddings = self.compute_embeddings()

        selected_target_indices = self.select_target(target_embeddings, target_budget)
        selected_source_indices = self.select_source(
            source_embeddings, target_embeddings[selected_target_indices], source_budget
        )

        self._last_selected_source_indices = selected_source_indices
        self._last_selected_target_indices = selected_target_indices
        self._cached_embeddings = (source_embeddings, target_embeddings)

        selected_source = self.source.select(selected_source_indices).add_column(
            "domain", ["source"] * len(selected_source_indices)
        )
        selected_target = self.target.select(selected_target_indices).add_column(
            "domain", ["target"] * len(selected_target_indices)
        )
        return concatenate_datasets([selected_source, selected_target])

    def generate_statistics(self, output_dir="dass_output/stats"):
        """Save selection statistics and plots. Call ``curate()`` first."""
        if self._last_selected_source_indices is None or self._last_selected_target_indices is None:
            raise RuntimeError("No selection has been made. Run `curate()` first.")

        stats_gen = StatisticsGenerator(
            source_dataset=self.source,
            target_dataset=self.target,
            source_embeddings=self._cached_embeddings[0],
            target_embeddings=self._cached_embeddings[1],
        )
        return stats_gen.generate(
            selected_source_indices=self._last_selected_source_indices,
            selected_target_indices=self._last_selected_target_indices,
            output_dir=output_dir,
            plot_utils=plot_utils,
        )
