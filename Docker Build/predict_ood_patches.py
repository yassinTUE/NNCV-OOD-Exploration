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

# Final chosen OOD settings
ENERGY_THRESHOLD = -1.122176   # ID vs combined(RA21, Far) intersection
TEMPERATURE = 1.0
INPUT_SIZE = (512, 512)

PATCH_WINDOW = 96
PATCH_STRIDE = PATCH_WINDOW // 2   # = 48
PATCH_METHOD = "patch_top5"


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


def extract_energy_map(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # logits: (1, C, H, W) -> energy_map: (H, W)
    energy_map = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy_map[0]


def extract_patch_scores(energy_map: torch.Tensor, window: int, stride: int) -> list[float]:
    h, w = energy_map.shape
    scores = []

    if window > h or window > w:
        return [energy_map.mean().item()]

    row_starts = list(range(0, h - window + 1, stride))
    col_starts = list(range(0, w - window + 1, stride))

    last_i = h - window
    last_j = w - window

    if row_starts[-1] != last_i:
        row_starts.append(last_i)
    if col_starts[-1] != last_j:
        col_starts.append(last_j)

    for i in row_starts:
        for j in col_starts:
            patch = energy_map[i:i + window, j:j + window]
            scores.append(patch.mean().item())

    return scores


def aggregate_patch_scores(patch_scores: list[float], method: str) -> float:
    if not patch_scores:
        raise ValueError("patch_scores is empty")

    patch_scores = sorted(patch_scores, reverse=True)
    n = len(patch_scores)

    if method == "patch_top1" or method == "patch_max":
        return patch_scores[0]

    if method == "patch_top3":
        k = min(3, n)
        return sum(patch_scores[:k]) / k

    if method == "patch_top5":
        k = min(5, n)
        return sum(patch_scores[:k]) / k

    if method == "patch_mean_all":
        return sum(patch_scores) / n

    raise ValueError(f"Unknown patch aggregation method: {method}")


def compute_image_patch_energy_score(
    logits: torch.Tensor,
    temperature: float,
    window: int,
    stride: int,
    method: str,
) -> float:
    energy_map = extract_energy_map(logits, temperature=temperature)
    patch_scores = extract_patch_scores(energy_map, window=window, stride=stride)
    return aggregate_patch_scores(patch_scores, method=method)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(n_classes=19)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # Keep the exact server-safe image discovery pattern from your working script
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

            patch_score = compute_image_patch_energy_score(
                logits=logits,
                temperature=TEMPERATURE,
                window=PATCH_WINDOW,
                stride=PATCH_STRIDE,
                method=PATCH_METHOD,
            )

            # Lower = ID, higher = OOD
            include_decision = patch_score < ENERGY_THRESHOLD

            predictions.append({
                "image_name": str(relative_path).replace("\\", "/"),
                "include": bool(include_decision),
            })

            print(
                f"[{img_path.name}] score={patch_score:.6f} "
                f"thr={ENERGY_THRESHOLD:.6f} "
                f"window={PATCH_WINDOW} stride={PATCH_STRIDE} method={PATCH_METHOD} -> "
                f"{'ID' if include_decision else 'OOD'}"
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "include"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")


if __name__ == "__main__":
    main()