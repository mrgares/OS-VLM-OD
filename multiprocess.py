import concurrent.futures
import fiftyone as fo
from tqdm import tqdm
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import multiprocessing
import torch

multiprocessing.set_start_method("spawn", force=True)

# Load dataset
dataset = fo.load_dataset("nuscenes")
print('Loaded dataset with %d samples' % len(dataset))

# Create filtered views for training and validation
train_view = dataset.match({"split": "train"})
val_view = dataset.match({"split": "validation"})

# Number of workers and batch size
workers = 3
batch_size = 5
dataset_name = "nuscenes"

samples = train_view.select_group_slices([
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", 
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
]).limit(60)


def limit_memory_fraction(memory_fraction=1/3):
    """
    Limits the memory fraction of the GPU to the specified fraction per worker.
    For a 16GB GPU and 3 workers, 1/3 sets roughly 5.3GB per worker.
    """
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

# Initialize a single GroundedSAM model per process to avoid reloading
def init_grounded_sam_model():
    return GroundedSAM(
        ontology=CaptionOntology({
            "person": "person",
            "ground vehicle car": "vehicle",
        }),
        box_threshold=0.5,
        text_threshold=0.5,
    )

# Define a function to process each sample using a specified groundedSAM model
def process_sample_batch(sample_ids, model, dataset_name):
    limit_memory_fraction(1/workers)
    torch.cuda.empty_cache()  
    
    dataset_ = fo.load_dataset(dataset_name)

    for sample_id in sample_ids:
        sample = dataset_[sample_id]
        results_SAM = model.predict(sample.filepath)

        detections = []
        for i, mask in enumerate(results_SAM.mask):
            height, width = mask.shape
            bb = results_SAM.xyxy[i]
            bb_normalized = [bb[0] / width, bb[1] / height, (bb[2] - bb[0]) / width, (bb[3] - bb[1]) / height]
            bb = bb.astype(int)
            mask_cropped = mask[bb[1]:bb[3], bb[0]:bb[2]]
            
            detection = fo.Detection(
                mask=mask_cropped,
                bounding_box=bb_normalized,
                confidence=round(results_SAM.confidence[i], 2),
                label=model.ontology.classes()[results_SAM.class_id[i]]
            )
            detections.append(detection)
        
        sample["pseudo_masks"] = fo.Detections(detections=detections)
        sample.save()

def process_batch_wrapper(sample_ids):
    model = init_grounded_sam_model()
    process_sample_batch(sample_ids, model, dataset_name)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        sample_ids = [sample.id for sample in samples]

        for i in tqdm(range(0, len(sample_ids), batch_size), desc="Scheduling batches"):
            batch = sample_ids[i:i + batch_size]
            futures.append(executor.submit(process_batch_wrapper, batch))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            future.result()  # Raises exceptions if any occurred during execution
