import csv
import math
from statistics import mean, pstdev

import matplotlib.pyplot as plt

CSV_PATH = "ood_compare_scores.csv"
OUT_PATH = "pdf_id_vs_avg_ra_far_energy_100.png"

TARGET_METHOD = "energy"
TARGET_TOP_PERCENT = 100
TARGET_EVAL_SPLIT = "all"


def gaussian_pdf(x, mu, sigma):
    if sigma <= 0:
        return 0.0
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def averaged_gaussian_pdf(x, mu_a, sigma_a, mu_b, sigma_b):
    return 0.5 * gaussian_pdf(x, mu_a, sigma_a) + 0.5 * gaussian_pdf(x, mu_b, sigma_b)


def fit_gaussian(scores):
    return mean(scores), pstdev(scores)


def find_intersection_numeric(mu1, sigma1, mu2, sigma2, x_min, x_max, steps=20000):
    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
    diffs = [gaussian_pdf(x, mu1, sigma1) - gaussian_pdf(x, mu2, sigma2) for x in xs]

    crossings = []
    for i in range(len(xs) - 1):
        d1 = diffs[i]
        d2 = diffs[i + 1]

        if d1 == 0:
            crossings.append(xs[i])
        elif d1 * d2 < 0:
            x1, x2 = xs[i], xs[i + 1]
            root = x1 - d1 * (x2 - x1) / (d2 - d1)
            crossings.append(root)

    if not crossings:
        return None

    lo = min(mu1, mu2)
    hi = max(mu1, mu2)
    between = [x for x in crossings if lo <= x <= hi]
    if between:
        return between[0]

    midpoint = 0.5 * (mu1 + mu2)
    return min(crossings, key=lambda x: abs(x - midpoint))


def find_intersection_with_average(mu_id, sigma_id, mu_ra, sigma_ra, mu_far, sigma_far, x_min, x_max, steps=20000):
    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
    diffs = [
        gaussian_pdf(x, mu_id, sigma_id) - averaged_gaussian_pdf(x, mu_ra, sigma_ra, mu_far, sigma_far)
        for x in xs
    ]

    crossings = []
    for i in range(len(xs) - 1):
        d1 = diffs[i]
        d2 = diffs[i + 1]

        if d1 == 0:
            crossings.append(xs[i])
        elif d1 * d2 < 0:
            x1, x2 = xs[i], xs[i + 1]
            root = x1 - d1 * (x2 - x1) / (d2 - d1)
            crossings.append(root)

    if not crossings:
        return None

    midpoint = (mu_id + 0.5 * (mu_ra + mu_far)) / 2.0
    return min(crossings, key=lambda x: abs(x - midpoint))


rows = []
with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (
            row["method"] == TARGET_METHOD
            and int(row["top_percent"]) == TARGET_TOP_PERCENT
            and row["eval_split"] == TARGET_EVAL_SPLIT
        ):
            rows.append(row)

scores = {
    "id": [],
    "near_ood_ra21": [],
    "far_ood": [],
}

for row in rows:
    split = row["split"]
    if split in scores:
        scores[split].append(float(row["score"]))

for split, vals in scores.items():
    if len(vals) < 2:
        raise RuntimeError(f"Not enough samples for split {split}")

mu_id, sigma_id = fit_gaussian(scores["id"])
mu_ra, sigma_ra = fit_gaussian(scores["near_ood_ra21"])
mu_far, sigma_far = fit_gaussian(scores["far_ood"])

all_vals = scores["id"] + scores["near_ood_ra21"] + scores["far_ood"]
x_min = min(all_vals) - 1.0
x_max = max(all_vals) + 1.0

thr_id_vs_ra = find_intersection_numeric(mu_id, sigma_id, mu_ra, sigma_ra, x_min, x_max)
thr_id_vs_avg = find_intersection_with_average(mu_id, sigma_id, mu_ra, sigma_ra, mu_far, sigma_far, x_min, x_max)

# build x-axis values without numpy
num_points = 1000
xs = [x_min + (x_max - x_min) * i / (num_points - 1) for i in range(num_points)]

id_pdf = [gaussian_pdf(x, mu_id, sigma_id) for x in xs]
ra_pdf = [gaussian_pdf(x, mu_ra, sigma_ra) for x in xs]
far_pdf = [gaussian_pdf(x, mu_far, sigma_far) for x in xs]
avg_pdf = [averaged_gaussian_pdf(x, mu_ra, sigma_ra, mu_far, sigma_far) for x in xs]

plt.figure(figsize=(10, 6))

plt.plot(xs, id_pdf, label=f"ID PDF (μ={mu_id:.3f}, σ={sigma_id:.3f})")
plt.plot(xs, ra_pdf, label=f"RA21 PDF (μ={mu_ra:.3f}, σ={sigma_ra:.3f})")
plt.plot(xs, far_pdf, label=f"Far OOD PDF (μ={mu_far:.3f}, σ={sigma_far:.3f})")
plt.plot(xs, avg_pdf, linestyle="--", linewidth=2, label="Average PDF of RA21 + Far OOD")

if thr_id_vs_ra is not None:
    plt.axvline(thr_id_vs_ra, linestyle=":", linewidth=2, label=f"ID vs RA21 intersection = {thr_id_vs_ra:.3f}")

if thr_id_vs_avg is not None:
    plt.axvline(thr_id_vs_avg, linestyle="-.", linewidth=2, label=f"ID vs avg(RA21,Far) intersection = {thr_id_vs_avg:.3f}")

plt.xlabel("Energy score")
plt.ylabel("PDF")
plt.title("ID vs RA21 / Far OOD Gaussian PDFs (energy, top=100%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved plot to {OUT_PATH}")
print(f"ID mean={mu_id:.6f}, std={sigma_id:.6f}")
print(f"RA21 mean={mu_ra:.6f}, std={sigma_ra:.6f}")
print(f"Far OOD mean={mu_far:.6f}, std={sigma_far:.6f}")
if thr_id_vs_ra is not None:
    print(f"ID vs RA21 intersection = {thr_id_vs_ra:.6f}")
if thr_id_vs_avg is not None:
    print(f"ID vs avg(RA21, FarOOD) intersection = {thr_id_vs_avg:.6f}")

# =========================
# ADDED BASELINE MSP BLOCK
# =========================

BASELINE_METHOD = "msp"
BASELINE_EVAL_SPLIT = "all"
BASELINE_OUT_PATH = "pdf_id_vs_avg_ra_far_msp_best.png"
BASELINE_SEP_PLOT_PATH = "msp_separation_vs_top_percent.png"


def compute_sep_score_from_scores(score_dict):
    mu_id_b, sigma_id_b = fit_gaussian(score_dict["id"])
    mu_ra_b, sigma_ra_b = fit_gaussian(score_dict["near_ood_ra21"])
    mu_far_b, sigma_far_b = fit_gaussian(score_dict["far_ood"])

    mu_mix_b = 0.5 * mu_ra_b + 0.5 * mu_far_b
    var_mix_b = (
        0.5 * (sigma_ra_b ** 2 + mu_ra_b ** 2)
        + 0.5 * (sigma_far_b ** 2 + mu_far_b ** 2)
        - mu_mix_b ** 2
    )
    sigma_mix_b = var_mix_b ** 0.5 if var_mix_b > 0 else 0.0

    raw_dist_b = abs(mu_id_b - mu_mix_b)
    denom_b = (sigma_id_b ** 2 + sigma_mix_b ** 2) ** 0.5
    sep_score_b = raw_dist_b / denom_b if denom_b > 0 else 0.0

    return {
        "mu_id": mu_id_b,
        "sigma_id": sigma_id_b,
        "mu_ra": mu_ra_b,
        "sigma_ra": sigma_ra_b,
        "mu_far": mu_far_b,
        "sigma_far": sigma_far_b,
        "mu_mix": mu_mix_b,
        "sigma_mix": sigma_mix_b,
        "raw_dist": raw_dist_b,
        "sep_score": sep_score_b,
    }


baseline_rows = []
with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["method"] == BASELINE_METHOD and row["eval_split"] == BASELINE_EVAL_SPLIT:
            baseline_rows.append(row)

candidate_top_percents = sorted(set(int(r["top_percent"]) for r in baseline_rows))

if not candidate_top_percents:
    raise RuntimeError(f"No candidate MSP settings found in {CSV_PATH}")

baseline_results = []

for top_percent in candidate_top_percents:
    baseline_scores = {
        "id": [],
        "near_ood_ra21": [],
        "far_ood": [],
    }

    for row in baseline_rows:
        if int(row["top_percent"]) == top_percent:
            split = row["split"]
            if split in baseline_scores:
                baseline_scores[split].append(float(row["score"]))

    valid = True
    for split_name, vals in baseline_scores.items():
        if len(vals) < 2:
            valid = False
            break

    if not valid:
        continue

    stats = compute_sep_score_from_scores(baseline_scores)
    baseline_results.append({
        "top_percent": top_percent,
        "scores": baseline_scores,
        **stats,
    })

if not baseline_results:
    raise RuntimeError("No valid MSP settings had enough samples for ID / RA21 / Far")

# Plot separation vs top percent for MSP
sep_xs = [r["top_percent"] for r in baseline_results]
sep_ys = [r["sep_score"] for r in baseline_results]

plt.figure(figsize=(8, 5))
plt.plot(sep_xs, sep_ys, marker="o")
plt.xlabel("Top percent")
plt.ylabel("Separation score")
plt.title("ID vs combined(RA21, Far) separation vs top percent (MSP)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASELINE_SEP_PLOT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved MSP separation plot to {BASELINE_SEP_PLOT_PATH}")

best_baseline = max(baseline_results, key=lambda r: r["sep_score"])

print("\nBest MSP setting by ID vs combined(RA21, Far) separation:")
print(
    f"top_percent={best_baseline['top_percent']}, "
    f"sep_score={best_baseline['sep_score']:.6f}, "
    f"raw_mean_distance={best_baseline['raw_dist']:.6f}, "
    f"id_mean={best_baseline['mu_id']:.6f}, "
    f"id_std={best_baseline['sigma_id']:.6f}, "
    f"ra_mean={best_baseline['mu_ra']:.6f}, "
    f"ra_std={best_baseline['sigma_ra']:.6f}, "
    f"far_mean={best_baseline['mu_far']:.6f}, "
    f"far_std={best_baseline['sigma_far']:.6f}, "
    f"mix_mean={best_baseline['mu_mix']:.6f}, "
    f"mix_std={best_baseline['sigma_mix']:.6f}"
)

baseline_scores = best_baseline["scores"]

mu_id_b = best_baseline["mu_id"]
sigma_id_b = best_baseline["sigma_id"]
mu_ra_b = best_baseline["mu_ra"]
sigma_ra_b = best_baseline["sigma_ra"]
mu_far_b = best_baseline["mu_far"]
sigma_far_b = best_baseline["sigma_far"]

all_vals_b = baseline_scores["id"] + baseline_scores["near_ood_ra21"] + baseline_scores["far_ood"]
x_min_b = min(all_vals_b) - 1.0
x_max_b = max(all_vals_b) + 1.0

thr_id_vs_ra_b = find_intersection_numeric(
    mu_id_b, sigma_id_b, mu_ra_b, sigma_ra_b, x_min_b, x_max_b
)
thr_id_vs_avg_b = find_intersection_with_average(
    mu_id_b, sigma_id_b, mu_ra_b, sigma_ra_b, mu_far_b, sigma_far_b, x_min_b, x_max_b
)

num_points_b = 1000
xs_b = [x_min_b + (x_max_b - x_min_b) * i / (num_points_b - 1) for i in range(num_points_b)]

id_pdf_b = [gaussian_pdf(x, mu_id_b, sigma_id_b) for x in xs_b]
ra_pdf_b = [gaussian_pdf(x, mu_ra_b, sigma_ra_b) for x in xs_b]
far_pdf_b = [gaussian_pdf(x, mu_far_b, sigma_far_b) for x in xs_b]
avg_pdf_b = [averaged_gaussian_pdf(x, mu_ra_b, sigma_ra_b, mu_far_b, sigma_far_b) for x in xs_b]

plt.figure(figsize=(10, 6))

plt.plot(xs_b, id_pdf_b, label=f"ID PDF (μ={mu_id_b:.3f}, σ={sigma_id_b:.3f})")
plt.plot(xs_b, ra_pdf_b, label=f"RA21 PDF (μ={mu_ra_b:.3f}, σ={sigma_ra_b:.3f})")
plt.plot(xs_b, far_pdf_b, label=f"Far OOD PDF (μ={mu_far_b:.3f}, σ={sigma_far_b:.3f})")
plt.plot(xs_b, avg_pdf_b, linestyle="--", linewidth=2, label="Average PDF of RA21 + Far OOD")

if thr_id_vs_ra_b is not None:
    plt.axvline(
        thr_id_vs_ra_b,
        linestyle=":",
        linewidth=2,
        label=f"ID vs RA21 intersection = {thr_id_vs_ra_b:.3f}"
    )

if thr_id_vs_avg_b is not None:
    plt.axvline(
        thr_id_vs_avg_b,
        linestyle="-.",
        linewidth=2,
        label=f"ID vs avg(RA21,Far) intersection = {thr_id_vs_avg_b:.3f}"
    )

plt.xlabel("MSP score")
plt.ylabel("PDF")
plt.title(f"ID vs RA21 / Far OOD Gaussian PDFs (MSP, best top={best_baseline['top_percent']}%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(BASELINE_OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved MSP Gaussian plot to {BASELINE_OUT_PATH}")
print(f"MSP ID mean={mu_id_b:.6f}, std={sigma_id_b:.6f}")
print(f"MSP RA21 mean={mu_ra_b:.6f}, std={sigma_ra_b:.6f}")
print(f"MSP Far OOD mean={mu_far_b:.6f}, std={sigma_far_b:.6f}")
if thr_id_vs_ra_b is not None:
    print(f"MSP ID vs RA21 intersection = {thr_id_vs_ra_b:.6f}")
if thr_id_vs_avg_b is not None:
    print(f"MSP ID vs avg(RA21, FarOOD) intersection = {thr_id_vs_avg_b:.6f}")