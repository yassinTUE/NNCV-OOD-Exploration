from pathlib import Path
import argparse
import csv

import torch
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model_segformer import Model


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)


def gather_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files = []
    for ext in exts:
        files.extend(folder.rglob(ext))
    return sorted(files)


def extract_energy_map(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    energy_map = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy_map[0]  # (H, W)


def extract_patch_scores(energy_map: torch.Tensor, window: int):
    stride = max(1, window // 2)
    h, w = energy_map.shape
    scores = []

    if window > h or window > w:
        return [energy_map.mean().item()], stride

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

    return scores, stride


def aggregate_patch_scores(patch_scores):
    patch_scores = sorted(patch_scores, reverse=True)
    n = len(patch_scores)

    def topk_mean(k):
        k = min(k, n)
        return sum(patch_scores[:k]) / k

    return {
        "patch_top1": topk_mean(1),
        "patch_top3": topk_mean(3),
        "patch_top5": topk_mean(5),
        "patch_top8": topk_mean(8),
        "patch_top10": topk_mean(10),
        "patch_top15": topk_mean(15),
        "patch_top20": topk_mean(20),
        "patch_top25": topk_mean(25),
        "patch_top30": topk_mean(30),
        "patch_top35": topk_mean(35),
        "patch_top40": topk_mean(40),
        "patch_top45": topk_mean(45),
        "patch_mean_all": sum(patch_scores) / n,
        "patch_max": patch_scores[0],
        "num_patches": n,
    }


def process_folder(model, folder: Path, split_name: str, label: int, device, temperature: float, patch_sizes):
    rows = []
    image_files = gather_images(folder)
    print(f"{split_name}: found {len(image_files)} images in {folder}")

    with torch.no_grad():
        for idx, img_path in enumerate(image_files, start=1):
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device)
            logits = model(x)
            energy_map = extract_energy_map(logits, temperature=temperature).cpu()

            for window in patch_sizes:
                patch_scores, stride = extract_patch_scores(energy_map, window)
                agg = aggregate_patch_scores(patch_scores)

                rows.append({
                    "filename": img_path.name,
                    "filepath": str(img_path),
                    "split": split_name,
                    "label": label,
                    "window": window,
                    "stride": stride,
                    "patch_top1": agg["patch_top1"],
                    "patch_top3": agg["patch_top3"],
                    "patch_top5": agg["patch_top5"],
                    "patch_top8": agg["patch_top8"],
                    "patch_top10": agg["patch_top10"],
                    "patch_top15": agg["patch_top15"],
                    "patch_top20": agg["patch_top20"],
                    "patch_top25": agg["patch_top25"],
                    "patch_top30": agg["patch_top30"],
                    "patch_top35": agg["patch_top35"],
                    "patch_top40": agg["patch_top40"],
                    "patch_top45": agg["patch_top45"],
                    "patch_mean_all": agg["patch_mean_all"],
                    "patch_max": agg["patch_max"],
                    "num_patches": agg["num_patches"],
                })

            if idx % 25 == 0 or idx == len(image_files):
                print(f"{split_name}: processed {idx}/{len(image_files)} images")

    return rows


def save_scores_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "filepath",
                "split",
                "label",
                "window",
                "stride",
                "patch_top1",
                "patch_top3",
                "patch_top5",
                "patch_top8",
                "patch_top10",
                "patch_top15",
                "patch_top20",
                "patch_top25",
                "patch_top30",
                "patch_top35",
                "patch_top40",
                "patch_top45",
                "patch_mean_all",
                "patch_max",
                "num_patches",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_stats(values):
    n = len(values)
    if n == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    mean_val = sum(values) / n
    if n > 1:
        var_val = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = var_val ** 0.5
    else:
        std_val = 0.0

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "min": min(values),
        "max": max(values),
    }


def save_summary_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    splits = ["id", "near_ood_ra21", "far_ood"]
    methods = [
        "patch_top1",
        "patch_top3",
        "patch_top5",
        "patch_top8",
        "patch_top10",
        "patch_top15",
        "patch_top20",
        "patch_top25",
        "patch_top30",
        "patch_top35",
        "patch_top40",
        "patch_top45",
        "patch_mean_all",
        "patch_max",
    ]

    summary_rows = []

    for window in sorted(set(r["window"] for r in rows)):
        for split in splits:
            split_rows = [r for r in rows if r["split"] == split and r["window"] == window]
            stride = split_rows[0]["stride"] if split_rows else max(1, window // 2)

            for method in methods:
                values = [r[method] for r in split_rows]
                stats = compute_stats(values)

                summary_rows.append({
                    "window": window,
                    "stride": stride,
                    "split": split,
                    "method": method,
                    "count": stats["count"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window",
                "stride",
                "split",
                "method",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--id-dir", type=str, required=True)
    parser.add_argument("--near-ood-dir", type=str, required=True)  # RA21 only
    parser.add_argument("--far-ood-dir", type=str, required=True)
    parser.add_argument("--out-scores-csv", type=str, default="ood_patch_scores.csv")
    parser.add_argument("--out-summary-csv", type=str, default="ood_patch_summary.csv")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Patch sizes: {args.patch_sizes}")
    print("Stride is always window // 2")

    model = Model(n_classes=19)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    all_rows = []
    all_rows += process_folder(
        model=model,
        folder=Path(args.id_dir),
        split_name="id",
        label=0,
        device=device,
        temperature=args.temperature,
        patch_sizes=args.patch_sizes,
    )
    all_rows += process_folder(
        model=model,
        folder=Path(args.near_ood_dir),
        split_name="near_ood_ra21",
        label=1,
        device=device,
        temperature=args.temperature,
        patch_sizes=args.patch_sizes,
    )
    all_rows += process_folder(
        model=model,
        folder=Path(args.far_ood_dir),
        split_name="far_ood",
        label=1,
        device=device,
        temperature=args.temperature,
        patch_sizes=args.patch_sizes,
    )

    save_scores_csv(all_rows, Path(args.out_scores_csv))
    save_summary_csv(all_rows, Path(args.out_summary_csv))

    print(f"Saved per-image scores to: {args.out_scores_csv}")
    print(f"Saved split summary to: {args.out_summary_csv}")
    print("Done. Post-process locally for Gaussian fitting and intersections.")


if __name__ == "__main__":
    main()