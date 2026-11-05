import os
import numpy as np
import pandas as pd
from .utils import compute_mmd

class StatisticsGenerator:
    def __init__(self, source_dataset, target_dataset, source_embeddings, target_embeddings):
        """
        Initialize with datasets and their corresponding embeddings.

        Args:
            source_dataset: List or dataset object for the source domain
            target_dataset: List or dataset object for the target domain
            source_embeddings: Numpy array of source embeddings
            target_embeddings: Numpy array of target embeddings
        """
        self.source = source_dataset
        self.target = target_dataset
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings

    def generate(
        self,
        selected_source_indices,
        selected_target_indices,
        output_dir="dass_statistics",
        visualization_methods=["tsne", "pca", "heatmap", "distribution"],
        save_plots=True,
        plot_utils=None
    ):
        """
        Generate and optionally visualize statistics.

        Args:
            selected_source_indices: List of indices from the source domain
            selected_target_indices: List of indices from the target domain
            output_dir: Directory to save statistics and plots
            visualization_methods: List of visualizations to generate
            save_plots: Whether to save plots
            plot_utils: Module or object with visualization functions (optional)
        Returns:
            A dictionary of statistics.
        """
        os.makedirs(output_dir, exist_ok=True)

        source_selected = self.source_embeddings[selected_source_indices]
        target_selected = self.target_embeddings[selected_target_indices]

        # Compute statistics
        stats = {
            'source_dataset_size': len(self.source),
            'target_dataset_size': len(self.target),
            'selected_source_size': len(selected_source_indices),
            'selected_target_size': len(selected_target_indices),
            'selection_ratio_source': len(selected_source_indices) / len(self.source),
            'selection_ratio_target': len(selected_target_indices) / len(self.target),
            "mmd": {
                "rbf_gamma_0.5": compute_mmd(source_selected, target_selected, kernel='rbf', gamma=0.5),
                "linear": compute_mmd(source_selected, target_selected, kernel='linear')
            }
        }

        # Compute mean embeddings
        source_mean = self.source_embeddings.mean(axis=0)
        target_mean = self.target_embeddings.mean(axis=0)
        selected_source_mean = self.source_embeddings[selected_source_indices].mean(axis=0)
        selected_target_mean = self.target_embeddings[selected_target_indices].mean(axis=0)

        # Inter-domain distance
        stats['source_target_distance'] = np.linalg.norm(source_mean - target_mean)
        stats['selected_source_target_distance'] = np.linalg.norm(
            selected_source_mean - selected_target_mean
        )

        # Intra-domain distances
        stats['source_intra_distance'] = self._intra_distance(self.source_embeddings, source_mean)
        stats['target_intra_distance'] = self._intra_distance(self.target_embeddings, target_mean)
        stats['selected_source_intra_distance'] = self._intra_distance(
            self.source_embeddings[selected_source_indices], selected_source_mean
        )
        stats['selected_target_intra_distance'] = self._intra_distance(
            self.target_embeddings[selected_target_indices], selected_target_mean
        )

        # Diversity
        stats['selected_source_diversity'] = self._diversity(self.source_embeddings, selected_source_indices)
        stats['selected_target_diversity'] = self._diversity(self.target_embeddings, selected_target_indices)

        # Save statistics
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, "selection_statistics.csv"), index=False)

        # Visualizations
        if save_plots and plot_utils:
            for method in visualization_methods:
                method = method.lower()
                if method in ["tsne", "pca"]:
                    plot_utils.visualize_embeddings(
                        self.source_embeddings,
                        self.target_embeddings,
                        selected_source_indices,
                        selected_target_indices,
                        method=method,
                        save_path=os.path.join(output_dir, f"{method}_visualization.png"),
                        title=f"Domain Adaptation Selection ({method.upper()})"
                    )
                elif method == "heatmap":
                    plot_utils.plot_similarity_heatmap(
                        self.source_embeddings,
                        self.target_embeddings,
                        selected_source_indices,
                        selected_target_indices,
                        save_path=os.path.join(output_dir, "similarity_heatmap.png")
                    )
                elif method == "distribution":
                    plot_utils.plot_domain_distribution(
                        self.source,
                        self.target,
                        selected_source_indices,
                        selected_target_indices,
                        save_path=os.path.join(output_dir, "domain_distribution.png")
                    )

        return stats

    def _intra_distance(self, embeddings, mean_embedding):
        return np.mean([np.linalg.norm(emb - mean_embedding) for emb in embeddings])

    def _diversity(self, embeddings, indices):
        if len(indices) <= 1:
            return 0.0
        pairs = [(i, j) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        return np.mean([
            np.linalg.norm(embeddings[indices[i]] - embeddings[indices[j]])
            for i, j in pairs
        ])
