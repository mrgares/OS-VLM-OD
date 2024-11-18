import fiftyone as fo
from fiftyone.types import FiftyOneDataset

# Define the directory where the exported dataset is located
export_dir = "/datastore/nuscenes_vlms/"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=export_dir,
    dataset_type=FiftyOneDataset,
    name = "imported_nuscenes",
    persistent=True
)

# Verify the dataset was loaded
print(dataset)
