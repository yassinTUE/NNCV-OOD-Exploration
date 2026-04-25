from pathlib import Path
import argparse
import csv

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)
from sklearn.metrics import roc_auc_score

from model_segformer import Model


TOP_PERCENTS = [
    0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
]


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


def extract_energy_flat(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    energy_map = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy_map.reshape(-1)


def extract_msp_flat(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    max_prob = probs.max(dim=1).values
    ood_score_map = 1.0 - max_prob
    return ood_score_map.reshape(-1)


def pooled_score(flat_scores: torch.Tensor, top_percent: float) -> float:
    k = max(1, int(top_percent * flat_scores.numel()))
    topk_vals = torch.topk(flat_scores, k=k, largest=True).values
    return topk_vals.mean().item()


def process_folder(model, folder: Path, split_name: str, label: int, device, temperature: float):
    rows = []
    image_files = gather_images(folder)
    print(f"{split_name}: found {len(image_files)} images in {folder}")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device)
            logits = model(x)

            energy_flat = extract_energy_flat(logits, temperature=temperature).cpu()
            msp_flat = extract_msp_flat(logits).cpu()

            rows.append({
                "filename": img_path.name,
                "filepath": str(img_path),
                "split": split_name,
                "label": label,
                "energy_flat": energy_flat,
                "msp_flat": msp_flat,
            })

    return rows


def find_best_threshold(rows, score_key="score"):
    scores = sorted(set(r[score_key] for r in rows))
    best = None

    for thr in scores:
        tp = tn = fp = fn = 0

        for r in rows:
            pred_ood = 1 if r[score_key] >= thr else 0
            true_ood = r["label"]

            if pred_ood == 1 and true_ood == 1:
                tp += 1
            elif pred_ood == 0 and true_ood == 0:
                tn += 1
            elif pred_ood == 1 and true_ood == 0:
                fp += 1
            else:
                fn += 1

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        if best is None or balanced_accuracy > best["balanced_accuracy"]:
            best = {
                "threshold": thr,
                "balanced_accuracy": balanced_accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

    return best


def compute_metrics(rows, threshold, score_key="score"):
    tp = tn = fp = fn = 0

    for r in rows:
        pred_ood = 1 if r[score_key] >= threshold else 0
        true_ood = r["label"]

        if pred_ood == 1 and true_ood == 1:
            tp += 1
        elif pred_ood == 0 and true_ood == 0:
            tn += 1
        elif pred_ood == 1 and true_ood == 0:
            fp += 1
        else:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    y_true = [r["label"] for r in rows]
    y_score = [r[score_key] for r in rows]

    if len(set(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)

    return {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": auroc,
    }


def save_scores_csv(all_rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "top_percent", "threshold_source", "eval_split",
                "filename", "filepath", "split", "label",
                "score", "threshold", "decision", "correct"
            ]
        )
        writer.writeheader()
        writer.writerows(all_rows)


def save_summary_csv(summary_rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "top_percent", "threshold_source", "eval_split",
                "threshold", "auroc", "balanced_accuracy",
                "sensitivity", "specificity", "tp", "tn", "fp", "fn"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--id-dir", type=str, required=True)
    parser.add_argument("--near-ood2-dir", type=str, required=True)
    parser.add_argument("--far-ood-dir", type=str, required=True)
    parser.add_argument("--out-scores-csv", type=str, default="ood_compare_scores.csv")
    parser.add_argument("--out-summary-csv", type=str, default="ood_compare_summary.csv")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(n_classes=19)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    base_rows = []
    base_rows += process_folder(model, Path(args.id_dir), "id", 0, device, args.temperature)
    base_rows += process_folder(model, Path(args.near_ood2_dir), "near_ood_ra21", 1, device, args.temperature)
    base_rows += process_folder(model, Path(args.far_ood_dir), "far_ood", 1, device, args.temperature)

    id_rows = [r for r in base_rows if r["split"] == "id"]
    ra21_rows = [r for r in base_rows if r["split"] == "near_ood_ra21"]
    far_rows = [r for r in base_rows if r["split"] == "far_ood"]

    threshold_sets = {
        "ra21": id_rows + ra21_rows,
        "all": id_rows + ra21_rows + far_rows,
    }

    eval_sets = {
        "ra21": id_rows + ra21_rows,
        "far": id_rows + far_rows,
        "all": id_rows + ra21_rows + far_rows,
    }

    methods = {
        "energy": "energy_flat",
        "msp": "msp_flat",
    }

    all_scored_rows = []
    summary_rows = []

    for method_name, flat_key in methods.items():
        for top_percent in TOP_PERCENTS:
            for threshold_source, threshold_rows_base in threshold_sets.items():
                threshold_rows = []
                for r in threshold_rows_base:
                    score = pooled_score(r[flat_key], top_percent)
                    threshold_rows.append({
                        "filename": r["filename"],
                        "filepath": r["filepath"],
                        "split": r["split"],
                        "label": r["label"],
                        "score": score,
                    })

                best = find_best_threshold(threshold_rows, score_key="score")

                for eval_name, eval_base_rows in eval_sets.items():
                    eval_rows = []
                    for r in eval_base_rows:
                        score = pooled_score(r[flat_key], top_percent)
                        decision = 1 if score >= best["threshold"] else 0
                        correct = 1 if decision == r["label"] else 0

                        eval_rows.append({
                            "filename": r["filename"],
                            "filepath": r["filepath"],
                            "split": r["split"],
                            "label": r["label"],
                            "score": score,
                        })

                        all_scored_rows.append({
                            "method": method_name,
                            "top_percent": int(top_percent * 100),
                            "threshold_source": threshold_source,
                            "eval_split": eval_name,
                            "filename": r["filename"],
                            "filepath": r["filepath"],
                            "split": r["split"],
                            "label": r["label"],
                            "score": score,
                            "threshold": best["threshold"],
                            "decision": decision,
                            "correct": correct,
                        })

                    metrics = compute_metrics(eval_rows, best["threshold"], score_key="score")
                    summary_rows.append({
                        "method": method_name,
                        "top_percent": int(top_percent * 100),
                        "threshold_source": threshold_source,
                        "eval_split": eval_name,
                        "threshold": metrics["threshold"],
                        "auroc": metrics["auroc"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                        "tp": metrics["tp"],
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                    })

                    print(
                        f"method={method_name:6s} | "
                        f"top={int(top_percent*100):3d}% | "
                        f"thr_src={threshold_source:13s} | "
                        f"eval={eval_name:13s} | "
                        f"auroc={metrics['auroc']:.4f} | "
                        f"bal_acc={metrics['balanced_accuracy']:.4f} | "
                        f"sens={metrics['sensitivity']:.4f} | "
                        f"spec={metrics['specificity']:.4f} | "
                        f"thr={metrics['threshold']:.4f}"
                    )

    save_scores_csv(all_scored_rows, Path(args.out_scores_csv))
    save_summary_csv(summary_rows, Path(args.out_summary_csv))

    print(f"Saved per-image scores to: {args.out_scores_csv}")
    print(f"Saved summary to: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
