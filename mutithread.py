# Number of workers and models to create
workers = 3

import os
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import fiftyone as fo
from autodistill.utils import plot
import cv2
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import queue
import torch

dataset_name = "nuscenes"
workers = 3

dataset = fo.load_dataset(dataset_name)
print('Loaded dataset with %d samples' % len(dataset))

# Create filtered views for training and validation
train_view = dataset.match({"split": "train"})
val_view = dataset.match({"split": "validation"})

print('Loaded dataset with %d samples' % len(dataset))
print('%d train samples' % len(train_view))
print('%d validation samples' % len(val_view))



samples = train_view.select_group_slices(["CAM_FRONT_LEFT", 
                                          "CAM_FRONT", 
                                          "CAM_FRONT_RIGHT", 
                                          "CAM_BACK_LEFT", 
                                          "CAM_BACK", 
                                          "CAM_BACK_RIGHT"]).limit(60)

model_queue = queue.Queue()

# Create multiple GroundedSAM models based on the number of workers
# Load multiple models and place them in the queue
for _ in range(workers):
    model_queue.put(
        GroundedSAM(
            ontology=CaptionOntology({
                "person": "person",
                "ground vehicle car": "vehicle",
            }),
            box_threshold=0.5,
            text_threshold=0.5,
        )
    )


######################################################
########## Code snippet for multithreading ##########
######################################################

def limit_memory_fraction(memory_fraction=1/3):
    """
    Limits the memory fraction of the GPU to the specified fraction per worker.
    For a 16GB GPU and 3 workers, 1/3 sets roughly 5.3GB per worker.
    """
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)



# Define a function to process each sample using a specified groundedSAM model
def process_sample(sample_id, dataset_name):
    # limit_memory_fraction(1/workers)
    # torch.cuda.empty_cache()  
    model = model_queue.get()
    try:
        # Load the sample and run the model
        dataset_ = fo.load_dataset(dataset_name)
        sample = dataset_[sample_id]
        results_SAM = model.predict(sample.filepath)
        
        detections = []
        for i, mask in enumerate(results_SAM.mask):
            height, width = mask.shape
            bb = results_SAM.xyxy[i]
            # Convert to [x, y, width, height] format and normalize
            bb_normalized = [bb[0] / width, bb[1] / height, (bb[2] - bb[0]) / width, (bb[3] - bb[1]) / height]
            
            # Crop the mask to the bounding box
            bb = bb.astype(int)
            mask_cropped = mask[bb[1]:bb[3], bb[0]:bb[2]]
            
            detection = fo.Detection(
                mask=mask_cropped, 
                bounding_box=bb_normalized,
                confidence=round(results_SAM.confidence[i], 2),
                label=model.ontology.classes()[results_SAM.class_id[i]]
            )
            detections.append(detection)
        
        # Save the detections to the sample
        sample["pseudo_masks"] = fo.Detections(detections=detections)
        sample.save()
    finally:
        model_queue.put(model) # Return the model to the queue
    

if __name__ == "__main__":
    # Parallelize processing across samples with two models
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_sample, sample.id, dataset_name)
            for sample in tqdm(samples, desc="Scheduling tasks")
        ]
        
        # Wait for all tasks to complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            future.result()  # This raises exceptions if any occurred during execution
