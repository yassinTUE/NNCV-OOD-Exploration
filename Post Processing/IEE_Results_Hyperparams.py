import os
import csv
import math
import statistics as stats
import matplotlib.pyplot as plt

# ---------- Path resolution ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_csv(filename):
    candidates = [
        os.path.join(SCRIPT_DIR, "data", filename),
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(os.path.dirname(SCRIPT_DIR), "Final assignment", "data", filename),
        os.path.join(os.path.dirname(SCRIPT_DIR), "Final assignment", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename}. Tried:\n" + "\n".join(candidates))

COMPARE_SCORES = resolve_csv("ood_compare_scores.csv")
PATCH_SUMMARY = resolve_csv("ood_patch_summary.csv")

# ---------- IEEE-ish font ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ---------- Helpers ----------
def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mean_std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None, None
    return stats.mean(vals), stats.pstdev(vals)

def separation_from_groups(id_scores, ra_scores, far_scores):
    mu_id, sigma_id = mean_std(id_scores)
    mu_ra, sigma_ra = mean_std(ra_scores)
    mu_far, sigma_far = mean_std(far_scores)
    if None in (mu_id, sigma_id, mu_ra, sigma_ra, mu_far, sigma_far):
        return None
    mu_mix = 0.5 * mu_ra + 0.5 * mu_far
    var_mix = (
        0.5 * (sigma_ra ** 2 + mu_ra ** 2)
        + 0.5 * (sigma_far ** 2 + mu_far ** 2)
        - mu_mix ** 2
    )
    sigma_mix = math.sqrt(max(var_mix, 0.0))
    denom = math.sqrt(sigma_id ** 2 + sigma_mix ** 2)
    if denom <= 0:
        return None
    return abs(mu_id - mu_mix) / denom

# ---------- Read compare scores ----------
with open(COMPARE_SCORES, "r", newline="", encoding="utf-8") as f:
    compare_rows = list(csv.DictReader(f))

# MSP top-percent vs separation
msp_rows = [r for r in compare_rows if r.get("method") == "msp" and r.get("eval_split") == "all"]
top_percents = sorted({int(r["top_percent"]) for r in msp_rows if r.get("top_percent")})

msp_x = []
msp_y = []

for tp in top_percents:
    subset = [r for r in msp_rows if int(r["top_percent"]) == tp]
    id_scores = [to_float(r["score"]) for r in subset if r.get("split") == "id"]
    ra_scores = [to_float(r["score"]) for r in subset if r.get("split") == "near_ood_ra21"]
    far_scores = [to_float(r["score"]) for r in subset if r.get("split") == "far_ood"]
    sep = separation_from_groups(id_scores, ra_scores, far_scores)
    if sep is not None:
        msp_x.append(tp)
        msp_y.append(sep)

# ---------- Read patch summary ----------
with open(PATCH_SUMMARY, "r", newline="", encoding="utf-8") as f:
    patch_rows = list(csv.DictReader(f))

for r in patch_rows:
    r["window"] = int(r["window"])
    r["mean"] = to_float(r["mean"])
    r["std"] = to_float(r["std"])

# keep only the 3 patch methods you want
patch_rows = [
    r for r in patch_rows
    if r["method"] in ["patch_top1", "patch_top3", "patch_mean_all"]
]

def sep_from_summary(rows3):
    by_split = {r["split"]: r for r in rows3}
    needed = {"id", "near_ood_ra21", "far_ood"}
    if not needed.issubset(by_split):
        return None
    mu_id, sigma_id = by_split["id"]["mean"], by_split["id"]["std"]
    mu_ra, sigma_ra = by_split["near_ood_ra21"]["mean"], by_split["near_ood_ra21"]["std"]
    mu_far, sigma_far = by_split["far_ood"]["mean"], by_split["far_ood"]["std"]
    if None in (mu_id, sigma_id, mu_ra, sigma_ra, mu_far, sigma_far):
        return None
    mu_mix = 0.5 * mu_ra + 0.5 * mu_far
    var_mix = (
        0.5 * (sigma_ra ** 2 + mu_ra ** 2)
        + 0.5 * (sigma_far ** 2 + mu_far ** 2)
        - mu_mix ** 2
    )
    sigma_mix = math.sqrt(max(var_mix, 0.0))
    denom = math.sqrt(sigma_id ** 2 + sigma_mix ** 2)
    if denom <= 0:
        return None
    return abs(mu_id - mu_mix) / denom

# Best separation by window
windows = sorted({r["window"] for r in patch_rows})
window_x = []
window_y = []

for w in windows:
    subset_w = [r for r in patch_rows if r["window"] == w]
    methods = sorted({r["method"] for r in subset_w})
    best_sep = None
    for method in methods:
        subset = [r for r in subset_w if r["method"] == method]
        sep = sep_from_summary(subset)
        if sep is not None and (best_sep is None or sep > best_sep):
            best_sep = sep
    if best_sep is not None:
        window_x.append(w)
        window_y.append(best_sep)

# Best separation by method, in the order top1 -> top3 -> mean-all
wanted_methods = ["patch_top1", "patch_top3", "patch_mean_all"]
method_labels_map = {
    "patch_top1": "Top-1",
    "patch_top3": "Top-3",
    "patch_mean_all": "Mean-all",
}

method_x = []
method_y = []
method_labels = []

for i, method in enumerate(wanted_methods):
    subset_m = [r for r in patch_rows if r["method"] == method]
    best_sep = None
    for w in sorted({r["window"] for r in subset_m}):
        subset = [r for r in subset_m if r["window"] == w]
        sep = sep_from_summary(subset)
        if sep is not None and (best_sep is None or sep > best_sep):
            best_sep = sep
    if best_sep is not None:
        method_x.append(i)
        method_y.append(best_sep)
        method_labels.append(method_labels_map[method])

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25))

# (a) MSP
axes[0].plot(msp_x, msp_y, marker="o", linewidth=1.0, markersize=3.0)
axes[0].set_xlabel("Top pixel percentage (%)")
axes[0].set_ylabel("Separation score")
axes[0].set_title("(a) MSP")
axes[0].grid(True, linewidth=0.35, alpha=0.4)

# (b) Window
axes[1].plot(window_x, window_y, marker="o", linewidth=1.0, markersize=3.0)
axes[1].set_xlabel("Patch window size")
axes[1].set_ylabel("Best separation score")
axes[1].set_title("(b) Energy window")
axes[1].grid(True, linewidth=0.35, alpha=0.4)

# (c) Patch aggregation
axes[2].plot(method_x, method_y, marker="o", linewidth=1.0, markersize=3.0)
axes[2].set_xticks(method_x)
axes[2].set_xticklabels(method_labels)
axes[2].set_xlabel("Patch aggregation")
axes[2].set_ylabel("Best separation score")
axes[2].set_title("(c) Energy aggregation")
axes[2].grid(True, linewidth=0.35, alpha=0.4)

fig.tight_layout()

png_path = os.path.join(SCRIPT_DIR, "results_hyperparameter_tuning_ieee.png")
pdf_path = os.path.join(SCRIPT_DIR, "results_hyperparameter_tuning_ieee.pdf")
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(png_path)
print(pdf_path)

if msp_x and msp_y:
    best_i = max(range(len(msp_y)), key=lambda i: msp_y[i])
    print(f"Best MSP top %: {msp_x[best_i]}")
if window_x and window_y:
    best_i = max(range(len(window_y)), key=lambda i: window_y[i])
    print(f"Best energy window: {window_x[best_i]}")
if method_x and method_y:
    best_i = max(range(len(method_y)), key=lambda i: method_y[i])
    print(f"Best selected patch aggregation: {method_labels[best_i]}")