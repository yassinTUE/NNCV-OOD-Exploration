# NNCV Semantic Segmentation + OOD Detection

This repository contains code used for a project on semantic segmentation and out-of-distribution (OOD) detection based on a SegFormer B2 model trained on Cityscapes, and using output logits, classifying an image as ID or OOD based on a confidence score. This Github enables reproducibility of results

## Repository contents

- `Model Training/`
  Final model training code and model definitions, coded as in paper.
- `Inference/`
  Evaluation scripts for image energy and MSP pixel OOD scoring and patch OOD scoring.
- `Post Processing/`
  Scripts for Gaussian fitting, hyperparameter analysis, and figure generation.
- `Docker Build/`
  Files for building your own Docker containers.
- `Data Download/`
  Helper scripts for downloading datasets used during development.
- `Checkpoints/`
  Stored checkpoint files.

## Main project components

- SegFormer model definition:
  `Model Training/model_segformer.py`
- Final SegFormer training script:
  `Model Training/train_Segfromer.py`
- Image level OOD analysis:
  `Inference/predict_ood.py`
- Patch/Window OOD analysis:
  `Inference/predict_ood_patches.py`
- Post-processing and plotting:
  `Post Processing/Gaussian_Fitting_Energy_and_MSP.py`
  `Post Processing/Gaussian_Fitting_PatchEnergy.py`
  `Post Processing/IEE_Results_Hyperparams.py`

## Important note on paths

This project was originally developed across both local and HPC environments.

As a result, some scripts use local or HPC specific paths and might require path cleanup before they run unchanged. In particular, data download, training, and inference job scripts may still assume the original development environment, and are not flexible for new enviroments.

Checkpoint handling was done in the HPC, while most other scripts were run locally during development.

## Reproducibility note

The core training, inference, and post-processing code is included here, but this repository should be treated as the project codebase rather than a fully cleaned final reproduction package. Before reuse on a new machine:

- dataset paths
- checkpoint paths
- Docker build paths
- any SLURM or HPC-specific launch scripts

## Status

The final method code is present in the repository, but some file paths are not yet fully standardized for a clean GitHub-only reproduction workflow.
