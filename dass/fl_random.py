from .dass import DASS


class FL_RANDOM(DASS):
    """Ablation: facility-location target selection (stage 1) + random source selection (stage 2)."""

    def select_source(self, source_embeddings, selected_target_embeddings, source_budget):
        return self._random_indices(len(source_embeddings), source_budget)
