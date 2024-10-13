# OS-VLM-OD
Off-the-shelf Visual Language Model for 2D Object Detection. In this repo we are going to try different VLMs to test their zero-shot capabilities.

## Setup
1. Clone the repo
2. Go to the repo directory
```bash
cd OS-VLM-OD
```
3. Create a docker image with the following command:
```bash
docker build -t vlm-od .
```
4. Run the docker container with the following command:
```bash
docker run --name vlm_od -it --gpus all -v /path/to/datastore:/path/to/datastore -v `pwd`:/workspace --shm-size=16g --network fiftyone_network -e FIFTYONE_DATABASE_URI=mongodb://fiftyone:27017 vlm-od 
```