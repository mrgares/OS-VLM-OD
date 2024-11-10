import torch
torch.backends.cudnn.benchmark = True 
import fiftyone as fo
from tqdm import tqdm
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# Load dataset
dataset = fo.load_dataset("nuscenes")
print('Loaded dataset with %d samples' % len(dataset))

# Create filtered views for training and validation
train_view = dataset.match({"split": "train"})
val_view = dataset.match({"split": "validation"})

dataset_name = "nuscenes"

samples = train_view.select_group_slices([
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", 
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
]).limit(3600)

groundedSAM_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "ground vehicle car": "vehicle",

        }
    ),
    box_threshold=0.5, 
    text_threshold=0.5,
)

for sample in tqdm(samples, desc="Processing samples", unit="sample"):  
    sample = dataset[sample.id]
    results_SAM = groundedSAM_model.predict(sample.filepath)
    
    detections = []
    for i, mask in enumerate(results_SAM.mask):
        height, width = results_SAM.mask[i].shape
        bb = results_SAM.xyxy[i]
        # Convert to [x, y, width, height] format and normalize
        bb_normalized = [bb[0]/width, bb[1]/height, (bb[2]-bb[0])/width, (bb[3]-bb[1])/height]
        
        # crop mask to the bounding box
        bb = bb.astype(int)
        mask_cropped = mask[bb[1]:bb[3], bb[0]:bb[2]]
        
        
        detection = fo.Detection(mask=mask_cropped, 
                                bounding_box=bb_normalized,
                                confidence=round(results_SAM.confidence[i], 2),
                                label=groundedSAM_model.ontology.classes()[results_SAM.class_id[i]])
        detections.append(detection)
    
    sample["pseudo_masks"] = fo.Detections(detections=detections)
    sample.save()