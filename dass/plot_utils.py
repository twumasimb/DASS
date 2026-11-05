import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def visualize_embeddings(
    source_embeddings,
    target_embeddings,
    selected_source_indices,
    selected_target_indices,
    method="tsne",
    save_path=None,
    title=None
):
    method = method.lower()
    assert method in ["tsne", "pca"], "method must be either 'tsne' or 'pca'"

    all_embeddings = np.vstack([source_embeddings, target_embeddings])
    labels = (
        ["source"] * len(source_embeddings)
        + ["target"] * len(target_embeddings)
    )

    # Highlight selected points
    total_source = len(source_embeddings)
    total_target = len(target_embeddings)

    highlights = np.zeros(len(all_embeddings))
    for i in selected_source_indices:
        highlights[i] = 1
    for j in selected_target_indices:
        highlights[total_source + j] = 1

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(reduced):
        color = "blue" if labels[i] == "source" else "green"
        marker = "o" if highlights[i] == 0 else "X"
        plt.scatter(x, y, color=color, marker=marker, alpha=0.7, s=50)

    plt.title(title or f"{method.upper()} visualization of embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_similarity_heatmap(
    source_embeddings,
    target_embeddings,
    selected_source_indices,
    selected_target_indices,
    save_path=None
):
    selected_source = source_embeddings[selected_source_indices]
    selected_target = target_embeddings[selected_target_indices]
    sim_matrix = cosine_similarity(selected_source, selected_target)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis", cbar_kws={"label": "Cosine Similarity"})
    plt.title("Cosine Similarity Heatmap: Selected Source vs Target Embeddings")
    plt.xlabel("Target")
    plt.ylabel("Source")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_domain_distribution(
    source_dataset,
    target_dataset,
    selected_source_indices,
    selected_target_indices,
    save_path=None
):
    source_total = len(source_dataset)
    target_total = len(target_dataset)
    source_selected = len(selected_source_indices)
    target_selected = len(selected_target_indices)

    plt.figure(figsize=(8, 5))
    categories = ["Source", "Target"]
    totals = [source_total, target_total]
    selected = [source_selected, target_selected]

    x = np.arange(len(categories))
    width = 0.4

    plt.bar(x - width / 2, totals, width, label="Total", color="lightgray")
    plt.bar(x + width / 2, selected, width, label="Selected", color="steelblue")
    plt.ylabel("Count")
    plt.title("Domain Data Distribution")
    plt.xticks(x, categories)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
