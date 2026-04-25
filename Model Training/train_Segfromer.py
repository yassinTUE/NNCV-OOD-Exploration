"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import random
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode, RandomCrop
import numpy as np

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from transformers import get_cosine_schedule_with_warmup
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode
)
import torch.nn.functional as F

# from model import Model
from model_segformer import Model

###### GPT CODE FOR VALIDATION LOSS DICE AND IOU
NUM_CLASSES = 19
IGNORE_INDEX = 255

def update_confmat(confmat, preds, labels):
    valid = labels != IGNORE_INDEX
    preds = preds[valid]
    labels = labels[valid]

    inds = labels * NUM_CLASSES + preds
    confmat += torch.bincount(
        inds,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def mean_iou_dice_from_confmat(confmat):
    confmat = confmat.float()

    tp = torch.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)

    return iou.mean().item(), dice.mean().item()

def dice_loss(logits, targets, num_classes=19, ignore_index=255, eps=1e-7):
    probs = torch.softmax(logits, dim=1)

    # create mask for valid pixels
    valid = targets != ignore_index

    probs = probs.permute(0, 2, 3, 1)[valid]   # (N_valid, C)
    targets = targets[valid]

    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

    intersection = (probs * targets_onehot).sum(dim=0)
    union = probs.sum(dim=0) + targets_onehot.sum(dim=0)

    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()
####### END GPT CODE


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size") ## Changed batch size to 8 for segformer from 64 of original
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate") ### Changed learning rate for fine tuning Segfromer from 0.001 to 
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser

class SegTrainTransform:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, img, mask):
        # random scale
        scale = 1.0 #scale = random.uniform(0.75, 1.5)
        new_h = int(img.height * scale)
        new_w = int(img.width * scale)

        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        # pad if needed
        pad_h = max(self.size[0] - new_h, 0)
        pad_w = max(self.size[1] - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)

        # random crop
        #i, j, h, w = RandomCrop.get_params(img, output_size=self.size)
        #img = TF.crop(img, i, j, h, w)
        #mask = TF.crop(mask, i, j, h, w)

        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # random horizontal flip
        #if random.random() < 0.5:
        if False:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # color jitter on image only
        #if random.random() < 0.8:
        if False:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)
            hue = random.uniform(-0.02, 0.02)

            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)

        # to tensor + normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        mask = torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0)

        return img, mask


class SegValTransform:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, img, mask):
        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        mask = torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0)

        return img, mask


class CityscapesSegWrapper(torch.utils.data.Dataset):
    def __init__(self, root, split, joint_transform):
        self.ds = Cityscapes(
            root,
            split=split,
            mode="fine",
            target_type="semantic",
        )
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img, mask = self.joint_transform(img, mask)
        return img, mask


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    img_transform = Compose([
    ToImage(),
    Resize((512, 512)),
    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    # Load the dataset and make a split for training and validation
    # Joint transforms for segmentation
    train_joint_transform = SegTrainTransform(size=(512, 512))
    valid_joint_transform = SegValTransform(size=(512, 512))

    # Load the dataset and make a split for training and validation
    train_dataset = CityscapesSegWrapper(
        args.data_dir,
        split="train",
        joint_transform=train_joint_transform,
    )

    valid_dataset = CityscapesSegWrapper(
        args.data_dir,
        split="val",
        joint_transform=valid_joint_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # # Define the model
    # model = Model(
    #    in_channels=3,  # RGB images
    #     n_classes=19,  # 19 classes in the Cityscapes dataset
    #).to(device)

    model = Model(n_classes=19).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)#, weight_decay=1e-4)

    # Define the scheduler for lr
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    best_valid_iou = -1.0
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            ce = criterion(outputs, labels)
            dice = dice_loss(outputs, labels, num_classes=19)

            loss = ce  #+ 0.1 * dice
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                update_confmat(confmat, preds, labels)
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            valid_mean_iou, valid_mean_dice = mean_iou_dice_from_confmat(confmat)
            wandb.log({
                "valid_loss": valid_loss,
                "valid_mean_iou": valid_mean_iou,
                "valid_mean_dice": valid_mean_dice,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_mean_iou > best_valid_iou:
                best_valid_iou = valid_mean_iou
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_iou={valid_mean_iou:.4f}-val_dice={valid_mean_dice:.4f}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_iou={valid_mean_iou:.4f}-val_dice={valid_mean_dice:.4f}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
