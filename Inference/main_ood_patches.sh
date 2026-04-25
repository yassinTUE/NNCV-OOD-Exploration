#!/bin/bash
python3 predict_ood_patches.py \
  --checkpoint /gpfs/home6/scur2432/NNCV/Final\ assignment/checkpoints/SegFormer-b2-training-basic/best_model-epoch=0014-val_iou=0.8030-val_dice=0.8843.pt \
  --id-dir ./data/cityscapes/leftImg8bit/val \
  --near-ood-dir /gpfs/home6/scur2432/NNCV/Final\ assignment/data/near-ood-roadanomaly21 \
  --far-ood-dir /gpfs/home6/scur2432/NNCV/Final\ assignment/data/far-ood \
  --out-scores-csv ood_patch_scores.csv \
  --out-summary-csv ood_patch_summary.csv \
  --patch-sizes 32 48 64 96 128
  