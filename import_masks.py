# import fiftyone as fo
# from tqdm import tqdm

# dataset = fo.load_dataset("nuscenes")
# dataset_imported = fo.load_dataset("imported_nuscenes")

# # set pseudo_masks field from dataset_imported to dataset based on the "sample_token" field
# for sample in tqdm(dataset_imported, desc="Importing masks...", unit=" samples"):
#     sample_token = sample["sample_token"]
#     sample_to_update = dataset.match({"sample_token":sample_token}).first()
#     sample_to_update["pseudo_masks"] = sample["pseudo_masks"]
#     sample_to_update.save()
#     print("Sample updated: %s" % sample_token)
#     break

import fiftyone as fo
from tqdm import tqdm

# Load datasets
dataset = fo.load_dataset("nuscenes")
dataset_imported = fo.load_dataset("imported_nuscenes")

print("Creating dictionary of pseudo masks...")
# Step 1: Create a dictionary with sample_token as key and pseudo_masks as value from dataset_imported
# Using dict comprehension for more efficient construction
pseudo_masks_dict = {sample["sample_token"]: sample["pseudo_masks"] for sample in tqdm(dataset_imported, desc="Collecting masks")}

# Convert the keys of pseudo_masks_dict to a set for faster lookup
pseudo_mask_tokens = set(pseudo_masks_dict.keys())

print("Preparing updates...")
# Step 2: Build updates dictionary directly, only for samples needing updates
updates = {}
for sample in tqdm(dataset, desc="Preparing updates"):
    sample_token = sample["sample_token"]
    if sample_token in pseudo_mask_tokens:
        updates[sample.id] = pseudo_masks_dict[sample_token]

print("Performing bulk update...")
# Step 3: Perform a bulk update on the dataset in one operation
dataset.set_values("pseudo_masks", updates, key_field="id")
print("Bulk update completed.")




