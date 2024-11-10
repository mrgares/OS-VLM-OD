import os
import torch
import fiftyone as fo
from tqdm import tqdm
from torch.multiprocessing import Process, set_start_method, Queue
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import gc

# Set multiprocessing to use 'spawn' start method
set_start_method('spawn', force=True)

def process_batch_on_gpu(gpu_id, sample_ids, result_queue):
    torch.cuda.set_device(gpu_id)  # Set GPU context for this process

    # Load dataset inside each process
    dataset = fo.load_dataset("nuscenes")

    # Initialize model on specified GPU
    groundedSAM_model = GroundedSAM(
        ontology=CaptionOntology(
            {
                "person": "human",
                "ground vehicle car": "vehicle",
            }
        ),
        box_threshold=0.5,
        text_threshold=0.5,
    )

    for sample_id in tqdm(sample_ids, desc=f"Processing on GPU {gpu_id}", unit="sample"):
        try:
            sample = dataset[sample_id]
            filepath = sample.filepath

            # Run prediction and process results
            result_SAM = groundedSAM_model.predict(filepath)
            detections = []
            for i, mask in enumerate(result_SAM.mask):
                height, width = mask.shape
                bb = result_SAM.xyxy[i]
                bb_normalized = [bb[0] / width, bb[1] / height, (bb[2] - bb[0]) / width, (bb[3] - bb[1]) / height]
                bb = bb.astype(int)
                mask_cropped = mask[bb[1]:bb[3], bb[0]:bb[2]]
                
                detection = fo.Detection(
                    mask=mask_cropped, 
                    bounding_box=bb_normalized,
                    confidence=round(result_SAM.confidence[i], 2),
                    label=groundedSAM_model.ontology.classes()[result_SAM.class_id[i]]
                )
                detections.append(detection)
            
            sample["pseudo_masks"] = fo.Detections(detections=detections)
            sample.save()

            # Free memory after processing each sample
            del result_SAM, detections, sample
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing sample {sample_id} on GPU {gpu_id}: {e}")

    # Add completion signal to the queue
    result_queue.put(f"Completed processing on GPU {gpu_id}")

if __name__ == "__main__":
    # Load dataset and prepare samples
    dataset = fo.load_dataset("nuscenes")
    print(f"Loaded dataset with {len(dataset)} groups")

    # Prepare the dataset and extract sample IDs for processing
    train_view = dataset.match({"split": "train"})
    print(f"Selecting training view with {len(train_view)} samples")
    samples = train_view.select_group_slices([
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", 
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
    ])
    print(f"Selected {len(samples)} samples for processing")
    sample_ids = samples.values("id")
    print(f"Selected {len(sample_ids)} samples for processing")
    # Divide samples across available GPUs
    num_gpus = torch.cuda.device_count()
    sample_batches = [
        sample_ids[i * len(sample_ids) // num_gpus : (i + 1) * len(sample_ids) // num_gpus]
        for i in range(num_gpus)
    ]
    print(f"Divided samples across {num_gpus} GPUs")    
    # Initialize a Queue to track completion of each GPU process
    result_queue = Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        print(f"Starting process on GPU {gpu_id}")
        process_sample_ids = sample_batches[gpu_id]
        p = Process(target=process_batch_on_gpu, args=(gpu_id, process_sample_ids, result_queue))
        processes.append(p)
        p.start()

    # Wait for each process to signal completion
    for _ in range(num_gpus):
        print(result_queue.get())

    # Ensure all processes finish
    for p in processes:
        p.join()
