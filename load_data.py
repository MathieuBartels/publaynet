from datasets import load_dataset
from tqdm import tqdm

print("Loading dataset...")
ds = load_dataset("jordanparker6/publaynet")

# Show first sample annotations
annotations = ds["train"][0]["annotations"]
print(f"\nFirst sample annotations: {annotations}")

# Collect unique category IDs with progress tracking
print("\nCollecting category IDs from all training samples...")
categories = set()
total_samples = len(ds["train"])

# Use tqdm for progress bar
for sample in tqdm(ds["train"], desc="Processing samples", total=total_samples, unit="sample"):
    annotations = sample["annotations"]
    for annotation in annotations:
        categories.add(annotation["category_id"])

print(f"\nUnique category IDs: {sorted(categories)}")
print(f"Total unique categories: {len(categories)}")


# for annotation in annotations:
#     print("--------------------------------")
#     print("annotation: ", annotation)
#     print("bbox: ", annotation["bbox"])
#     print("id: ", annotation["id"])
#     print("image_id: ", annotation["image_id"])
#     print("segmentation: ", annotation["segmentation"])
#     print("category_id: ", annotation["category_id"])
#     print("iscrowd: ", annotation["iscrowd"])
#     print("area: ", annotation["area"])
#     categories.append(annotation["category_id"])
#     print("--------------------------------")
