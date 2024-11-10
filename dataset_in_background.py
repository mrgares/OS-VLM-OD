import fiftyone as fo
import time
# Replace `your_dataset` with your actual dataset loading code
dataset = fo.load_dataset("nuscenes")

# Launch FiftyOne app in remote mode
session = fo.launch_app(dataset)

# Keep the session alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")

