# SharkVision: Exploring deep learning for shark species classification from BRUVS videos

![cover image of SharkVision project, showing a shark, project title and affiliate logos](sharkvision.png)

## Project Summary

This repository contains the code for the SharkVision project - my take on a group project completed as part of the Intelligent Earth UKRI CDT in AI for the Environment training programme. For this project, my collaborators and I were tasked with building a computer vision pipeline for processing Baited Remote Underwater Video Systems (BRUVS) videos expected to contain four different shark species. The goal was to help researchers go from raw videos to specific video frames classified as containing one or more target species.

To do this, we implemented and compared three different modelling approaches: a ResNet50 baseline, an approach based on Contrastive Language–Image Pretraining (CLIP), and an approach using a frozen DinoV2 backbone with a hyperparameter-tuned linear probe as the classifier head.

We found that our DINO model achieved an out-of-sample accuracy of 90.8%, outperforming all other approaches. For more details on the methodology and results (and some pretty niche memes), check out the project presentation here.

## Data

Due to this being an ongoing research project, and due to the sensitive nature of the data due to shark conservation concerns, the raw data for this project is not public yet. The code can be adapted for any input data - feel free to reach out with questions on this topic.

## Workflow

1. Clone this repository locally and set it as your current directory.
2. Create a mamba environment using the `shark-environment.yml` file: 
```
mamba env create -f shark-environment.yml
mamba activate shark-env
```

3. Inside your current directory (which should be `iecdt-shark-id`) clone the following repositories: 
- https://github.com/OlgaIsupova/IEarth_CDT_shark_detection/tree/main
- https://github.com/filippovarini/sharktrack/tree/master
4. Follow the SharkTrack repository instructions and run the software on your shark videos (ensuring to save the output into `./data/cropped_sharks`).
5. If you want to label any of your own data, run `python ./data_processing/image_labelling.py` and ensure the labelled CSV is getting saved into `./data`. This will open the first image in your folder. All you have to do to label the image currently on your screen is press one of the following keys: 

```
=== Shark Species Labeling ===
Press:
  1: grey_reef_shark
  2: blacktip_reef_shark
  3: whitetip_reef_shark
  4: tawny_nurse_shark
  0: unclear/other
  q: quit
  s: skip
  b: go back
  ESC: save and quit
========================================
```
If you wish to customise the labels, this can be done by editing the `image_labelling.py` script.

6. Run any of the models from the `models` folder on your data, either as they come or fine-tuned on your newly labelled data.

## Hyperparameters

For the DINO model, we ran a GridSearch to determine that the following hyperparameters maximised performance: 

- learning_rate: 0.0001
- batch_size: 32
- hidden_dim: 256
- dropout: 0.3
- optimizer: adamw 
- weight_decay: 0.0001

## Contents

In the `data_processing` folder you will find:

- `crop_sharks.py`: takes SharkTrack output, and crops shark detection bounding boxes as individual images.
- `image_labelling.py`: image labelling software script (takes any folder of images as input, and outputs a CSV with image file paths and custom labels)
- `data_splitting.py`: code for producing the fine-tuning and test label datasets.

In the `models` folder, you will find one script per model (one for ResNet50, one for CLIP, one for the DINOv2 model and one for the tuned version of the DINOv2 model). The scripts train and evaluate the models on the test set (and fine-tune / provide few-shot examples where this is applicable). 

The `results` folder has the saved best models and parameters from the tuning process.

## Tech

This project was executed across three compute environments: a MacBook Pro (Apple M3 Pro) for local development, a JASMIN scientific analysis server (sci-vm-04) for CPU-based processing, and a JASMIN Orchid GPU cluster node (Nvidia A100) for model training, tuning and evaluation.

## Contact

For any questions, email Tedi Yankov (teodor.yankov@new.ox.ac.uk). 
