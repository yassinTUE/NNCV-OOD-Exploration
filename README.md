# NNCV Semantic Segmentation and OOD Detection

This repository contains the code for an NNCV project on semantic segmentation and out-of-distribution (OOD) detection using a SegFormer backbone trained on Cityscapes.

The repository includes:

- SegFormer training code for semantic segmentation
- image-level OOD scoring with energy and MSP
- patch-based energy OOD scoring
- post-processing scripts for Gaussian fitting and hyperparameter analysis
- Docker-side prediction scripts used for challenge-style submission

## Repository structure

- `Model Training/`
  Training scripts and model definitions
- `Inference/`
  Evaluation scripts for OOD scoring
- `Post Processing/`
  Plotting and result analysis scripts
- `Docker Build/`
  Submission-related Docker and prediction files
- `Data Download/`
  Data download helper scripts
- `Checkpoints/`
  Model checkpoint storage

## 1. Installation

This project was developed with Python and PyTorch. The code imports the following main packages:

- `torch`
- `torchvision`
- `transformers`
- `timm`
- `Pillow`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `wandb`

Example installation:

```bash
pip install torch torchvision
pip install transformers timm Pillow numpy matplotlib scikit-learn wandb
```

If running on GPU or on an HPC cluster, install the PyTorch build that matches the local CUDA environment.

## 2. Data preparation

The project uses Cityscapes for semantic segmentation and additional OOD image sets for evaluation.

Expected datasets:

- Cityscapes
- Near-OOD set
- RoadAnomaly21
- Far-OOD set

Data download helper scripts are provided in `Data Download/`:

- `download_docker_and_data.sh`
- `download_far_ood_png.sh`
- `download_near_ood_ra21.sh`

These scripts were originally used in an HPC workflow. Before running them on a new machine, check the output paths and environment assumptions.

For local reproduction, make sure the datasets are available in paths matching the commands used below, or update the paths accordingly.

Typical structure expected by the Python scripts:

```text
data/
  cityscapes/
  near-ood/
  near-ood-roadanomaly21/
  far-ood/
```

## 3. Training

The final segmentation model is the SegFormer model defined in:

- `Model Training/model_segformer.py`

The main training script is:

- `Model Training/train_Segfromer.py`

Example training command:

```bash
python "Model Training/train_Segfromer.py" \
  --data-dir ./data/cityscapes \
  --batch-size 8 \
  --epochs 15 \
  --lr 5e-5 \
  --num-workers 10 \
  --seed 42 \
  --experiment-id "SegFormer-b2-training-basic"
```

Notes:

- training logs are sent to Weights and Biases through `wandb`
- checkpoints are written into `checkpoints/<experiment-id>/`
- the repository also contains older U-Net files, but the final method is the SegFormer pipeline above

## 4. Image-level OOD evaluation

The image-level OOD evaluation script is:

- `Inference/predict_ood.py`

This script computes:

- energy-based scores
- MSP baseline scores
- summary CSV files with AUROC, balanced accuracy, sensitivity, and specificity

Example command:

```bash
python "Inference/predict_ood.py" \
  --checkpoint ./Checkpoints/model.pt \
  --id-dir ./data/cityscapes/leftImg8bit/val \
  --near-ood-dir ./data/near-ood \
  --near-ood2-dir ./data/near-ood-roadanomaly21 \
  --far-ood-dir ./data/far-ood \
  --out-scores-csv ood_compare_scores.csv \
  --out-summary-csv ood_compare_summary.csv
```

Outputs:

- `ood_compare_scores.csv`
- `ood_compare_summary.csv`

## 5. Patch-based OOD evaluation

The patch-based OOD evaluation script is:

- `Inference/predict_ood_patches.py`

Example command:

```bash
python "Inference/predict_ood_patches.py" \
  --checkpoint ./Checkpoints/model.pt \
  --id-dir ./data/cityscapes/leftImg8bit/val \
  --near-ood-dir ./data/near-ood-roadanomaly21 \
  --far-ood-dir ./data/far-ood \
  --out-scores-csv ood_patch_scores.csv \
  --out-summary-csv ood_patch_summary.csv \
  --patch-sizes 32 48 64 96 128
```

Outputs:

- `ood_patch_scores.csv`
- `ood_patch_summary.csv`

## 6. Post-processing and figures

The repository includes the following result analysis scripts:

- `Post Processing/Gaussian_Fitting_Energy_and_MSP.py`
- `Post Processing/Gaussian_Fitting_PatchEnergy.py`
- `Post Processing/IEE_Results_Hyperparams.py`

These scripts use the CSV outputs from the inference stage to generate:

- Gaussian distribution plots
- threshold intersection plots
- MSP hyperparameter analysis
- patch window and aggregation analysis

Typical usage:

```bash
python "Post Processing/Gaussian_Fitting_Energy_and_MSP.py"
python "Post Processing/Gaussian_Fitting_PatchEnergy.py"
python "Post Processing/IEE_Results_Hyperparams.py"
```

Before running them, make sure the required CSV files exist in the working directory:

- `ood_compare_scores.csv`
- `ood_patch_summary.csv`

## 7. Submission-related files

The `Docker Build/` folder contains challenge-style prediction scripts:

- `predict_Segmentation_Map.py`
- `predict_ood_energy.py`
- `predict_ood_MSP.py`
- `predict_ood_patches.py`

These scripts are written for a container environment using:

- input directory: `/data`
- output directory: `/output`

They also assume a model checkpoint at:

- `/app/model.pt`

If reproducing the submission container locally, verify that the Docker build context contains the expected model files and paths.

## 8. Important reproducibility note

This repository was developed across both local and HPC environments.

Some shell scripts still contain older HPC-specific or machine-specific paths. The core Python code for training, inference, and post-processing is included here, but a few wrapper scripts may need path cleanup before they run unchanged on another machine.

In particular, verify:

- dataset paths
- checkpoint paths
- Docker build paths
- any SLURM or HPC launch scripts

For reproduction, it is recommended to use the Python commands in this README directly rather than relying on the shell wrapper scripts without inspection.

## 9. Reproducing the main pipeline

To reproduce the main workflow from this repository:

1. Install the dependencies.
2. Download and place the datasets in the expected local folders.
3. Train the SegFormer model using `Model Training/train_Segfromer.py`, or use an existing checkpoint.
4. Run `Inference/predict_ood.py` for image-level OOD analysis.
5. Run `Inference/predict_ood_patches.py` for patch-based OOD analysis.
6. Run the scripts in `Post Processing/` to generate the plots and hyperparameter figures.

## 10. Final note

The final method code is included in this repository, but some paths are not yet fully standardized for a completely portable GitHub-only reproduction workflow. The README therefore documents the intended commands and required inputs as clearly as possible, while noting where manual path verification may still be needed.
