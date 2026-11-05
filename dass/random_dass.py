from .dass import DASS


class RANDOM_DASS(DASS):
    """Ablation: random target selection (stage 1) + SMI source selection (stage 2)."""

    def select_target(self, target_embeddings, target_budget):
        return self._random_indices(len(target_embeddings), target_budget)
