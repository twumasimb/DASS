"""Minimal DASS usage example on public HuggingFace datasets (IMDB -> SST-2).

Runs only the data-selection step (no fine-tuning), so it finishes in a few minutes.
"""

from datasets import Dataset, load_dataset

from dass import DASS


def to_input_label(dataset, text_col):
    return Dataset.from_dict({"input": dataset[text_col], "label": dataset["label"]})


# Source: movie reviews (labeled, plentiful). Target: short movie-review sentences (low-resource).
source = to_input_label(load_dataset("imdb", split="train").shuffle(seed=0).select(range(1000)), "text")
target = to_input_label(load_dataset("stanfordnlp/sst2", split="train").select(range(500)), "sentence")

dass = DASS(
    source=source,
    target=target,
    model="all-MiniLM-L6-v2",
    budget=100,
    target_ratio=0.3,      # 30 target points to annotate + 70 source points
    source_method="fl1mi",
    source_eta=0.8,
)

selected = dass.curate()
print(selected)
print(selected.to_pandas()["domain"].value_counts())

stats = dass.generate_statistics(output_dir="example_output/stats")
print({k: v for k, v in stats.items() if k != "mmd"})
