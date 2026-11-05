"""Build a 1,000-review subset per Amazon category (first 1,000 reviews, unbalanced).

Run from the data/ folder:  python convert_to_hf_data.py
"""
import json
from datasets import Dataset, DatasetDict

def convert_to_huggingface_dataset(json_file_path):
    """
    Converts a JSON lines file into a Hugging Face Dataset.
    
    Each entry must have 'reviewText' and 'overall' fields.
    'reviewText' becomes the input, and 'label' is 1 if overall > 3, else 0.
    
    Args:
        json_file_path (str): Path to the JSON file (one JSON object per line).
        
    Returns:
        Dataset: Hugging Face Dataset with 'input' and 'label' columns.
    """
    processed_data = []

    with open(json_file_path, 'r') as file:
        for line in file:
            try:
                review = json.loads(line.strip())
                if 'reviewText' in review and 'overall' in review:
                    processed_data.append({
                        'input': review['reviewText'],
                        'label': int(review['overall'] > 3.0)
                    })
            except json.JSONDecodeError:
                continue  # skip malformed lines

    return Dataset.from_list(processed_data)

def build_dataset_dict(names_and_paths, sample_size=1000):
    """
    Builds a DatasetDict from multiple JSON paths.
    
    Args:
        names_and_paths (list): List of tuples (name, path) for each dataset.
        sample_size (int): Number of examples to keep per dataset.
    
    Returns:
        DatasetDict: A dictionary of Hugging Face Datasets.
    """
    return DatasetDict({
        name: convert_to_huggingface_dataset(path).select(range(sample_size))
        for name, path in names_and_paths
    })

if __name__ == "__main__":
    dataset_configs = [
        ('Fashion', 'raw/AMAZON_FASHION.json'),
        ('Beauty', 'raw/All_Beauty.json'),
        ('Appliances', 'raw/Appliances.json'),
        ('Magazine', 'raw/Magazine_Subscriptions.json')
    ]

    dataset_dict = build_dataset_dict(dataset_configs, sample_size=1000)
    dataset_dict.save_to_disk('processed/final_dataset')
