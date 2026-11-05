"""Build a label-balanced, de-duplicated 1,000-review subset per Amazon category.

Run from the data/ folder:  python create_balanced_dataset.py
"""
import json
from datasets import Dataset, DatasetDict
import random

def convert_to_balanced_deduplicated_dataset(json_file_path, target_size=1000):
    data = []

    with open(json_file_path, 'r') as file:
        for line in file:
            review = json.loads(line.strip())
            if 'reviewText' in review and 'overall' in review:
                input_text = review['reviewText'].strip()
                if input_text:
                    label = 1 if review['overall'] > 3.0 else 0
                    data.append({'input': input_text, 'label': label})

    # Remove duplicates based on 'input'
    seen = set()
    unique_data = []
    for item in data:
        if item['input'] not in seen:
            seen.add(item['input'])
            unique_data.append(item)

    # Separate and balance labels
    label_0 = [item for item in unique_data if item['label'] == 0]
    label_1 = [item for item in unique_data if item['label'] == 1]
    min_len = min(len(label_0), len(label_1), target_size // 2)

    # Select equal number from both labels
    balanced = random.sample(label_0, min_len) + random.sample(label_1, min_len)
    random.shuffle(balanced)

    # If we have more than target_size, truncate
    if len(balanced) > target_size:
        balanced = random.sample(balanced, target_size)

    return Dataset.from_list(balanced)

if __name__ == "__main__":
    random.seed(42)

    # Dataset paths
    names_and_paths = [
        ('Fashion', 'raw/AMAZON_FASHION.json'),
        ('Beauty', 'raw/All_Beauty.json'),
        ('Appliances', 'raw/Appliances.json'),
        ('Magazine', 'raw/Magazine_Subscriptions.json')
    ]

    # Build dataset dictionary
    dataset_dict = DatasetDict()
    for name, path in names_and_paths:
        dataset_dict[name] = convert_to_balanced_deduplicated_dataset(path, target_size=1000)

    # Save
    dataset_dict.save_to_disk('processed/final_dataset_balanced')
