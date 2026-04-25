from pathlib import Path
import os
import csv

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model import Model

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

ENERGY_THRESHOLD = -0.966869
TEMPERATURE = 1.0
TOP_PERCENT = 1.0
INPUT_SIZE = (512, 512)


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)


def postprocess(logits: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    pred = torch.argmax(logits, dim=1, keepdim=True).float()
    pred = F.interpolate(pred, size=original_shape, mode="nearest")
    pred = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def compute_image_energy_score(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_percent: float = 0.90,
) -> float:
    energy_map = -temperature * torch.logsumexp(logits / temperature, dim=1)
    flat = energy_map.reshape(-1)
    k = max(1, int(top_percent * flat.numel()))
    topk_vals = torch.topk(flat, k=k, largest=True).values
    return topk_vals.mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(n_classes=19)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = sorted(Path(IMAGE_DIR).glob("**/*.png"))
    print(f"Found {len(image_files)} images to process.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            x = preprocess(img).to(device)
            logits = model(x)

            seg_pred = postprocess(logits, original_shape)

            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

            energy_score = compute_image_energy_score(
                logits,
                temperature=TEMPERATURE,
                top_percent=TOP_PERCENT,
            )
            include_decision = energy_score < ENERGY_THRESHOLD

            predictions.append({
                "image_name": str(relative_path).replace("\\", "/"),
                "include": bool(include_decision),
            })

            print(
                f"[{img_path.name}] energy={energy_score:.4f} "
                f"thr={ENERGY_THRESHOLD:.4f} top={TOP_PERCENT:.2f} -> "
                f"{'ID' if include_decision else 'OOD'}"
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "include"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")


if __name__ == "__main__":
    main()