import csv
import math
import matplotlib.pyplot as plt


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
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.35,
})

CSV_PATH = "ood_patch_summary.csv"
OUT_PREFIX = "patch_means"

METHODS = [
    ("patch_top1", "Top 1 patch mean"),
    ("patch_top3", "Top 3 patch mean"),
    ("patch_top5", "Top 5 patch mean"),
    ("patch_top8", "Top 8 patch mean"),
    ("patch_top10", "Top 10 patch mean"),
    ("patch_top15", "Top 15 patch mean"),
    ("patch_top20", "Top 20 patch mean"),
    ("patch_top25", "Top 25 patch mean"),
    ("patch_top30", "Top 30 patch mean"),
    ("patch_top35", "Top 35 patch mean"),
    ("patch_top40", "Top 40 patch mean"),
    ("patch_top45", "Top 45 patch mean"),
    ("patch_mean_all", "Mean of all patches"),
]

TOP_METHODS_ONLY = [
    ("patch_top1", "Top 1"),
    ("patch_top3", "Top 3"),
    ("patch_top5", "Top 5"),
    ("patch_top8", "Top 8"),
    ("patch_top10", "Top 10"),
    ("patch_top15", "Top 15"),
    ("patch_top20", "Top 20"),
    ("patch_top25", "Top 25"),
    ("patch_top30", "Top 30"),
    ("patch_top35", "Top 35"),
    ("patch_top40", "Top 40"),
    ("patch_top45", "Top 45"),
]

SPLITS = [
    ("id", "ID"),
    ("near_ood_ra21", "RA21"),
    ("far_ood", "Far OOD"),
]


def read_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["window"] = int(row["window"])
            row["stride"] = int(row["stride"])
            row["count"] = int(row["count"])
            row["mean"] = float(row["mean"])
            row["std"] = float(row["std"])
            row["min"] = float(row["min"])
            row["max"] = float(row["max"])
            rows.append(row)
    return rows


def get_series(rows, method_key, split_key):
    selected = [
        r for r in rows
        if r["method"] == method_key and r["split"] == split_key
    ]
    selected.sort(key=lambda r: r["window"])
    xs = [r["window"] for r in selected]
    ys = [r["mean"] for r in selected]
    return xs, ys


def make_plot(rows, method_key, method_title):
    plt.figure(figsize=(8, 5))

    for split_key, split_label in SPLITS:
        xs, ys = get_series(rows, method_key, split_key)
        if xs:
            plt.plot(xs, ys, marker="o", label=split_label)

    plt.xlabel("Window size")
    plt.ylabel("Mean score")
    plt.title(method_title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = f"{OUT_PREFIX}_{method_key}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def get_stat(rows, window, method_key, split_key, stat_key):
    for r in rows:
        if r["window"] == window and r["method"] == method_key and r["split"] == split_key:
            return r[stat_key]
    return None


def compute_sep_score(rows, window, method_key):
    mu_id = get_stat(rows, window, method_key, "id", "mean")
    std_id = get_stat(rows, window, method_key, "id", "std")

    mu_ra = get_stat(rows, window, method_key, "near_ood_ra21", "mean")
    std_ra = get_stat(rows, window, method_key, "near_ood_ra21", "std")

    mu_far = get_stat(rows, window, method_key, "far_ood", "mean")
    std_far = get_stat(rows, window, method_key, "far_ood", "std")

    if None in (mu_id, std_id, mu_ra, std_ra, mu_far, std_far):
        return None

    mu_mix = 0.5 * mu_ra + 0.5 * mu_far
    var_mix = (
        0.5 * (std_ra ** 2 + mu_ra ** 2)
        + 0.5 * (std_far ** 2 + mu_far ** 2)
        - mu_mix ** 2
    )
    std_mix = var_mix ** 0.5 if var_mix > 0 else 0.0

    raw_dist = abs(mu_id - mu_mix)
    denom = (std_id ** 2 + std_mix ** 2) ** 0.5
    sep_score = raw_dist / denom if denom > 0 else 0.0

    return {
        "window": window,
        "method": method_key,
        "score": sep_score,
        "raw_dist": raw_dist,
        "mu_id": mu_id,
        "std_id": std_id,
        "mu_ra": mu_ra,
        "std_ra": std_ra,
        "mu_far": mu_far,
        "std_far": std_far,
        "mu_mix": mu_mix,
        "std_mix": std_mix,
    }


def make_separation_vs_window_plot(rows, method_key, method_title):
    all_windows = sorted(set(r["window"] for r in rows))
    xs = []
    ys = []

    for window in all_windows:
        result = compute_sep_score(rows, window, method_key)
        if result is not None:
            xs.append(window)
            ys.append(result["score"])

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Window size")
    plt.ylabel("Separation score")
    plt.title(f"ID vs combined(RA21, Far) separation vs window\n{method_title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"separation_vs_window_{method_key}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def make_separation_vs_method_plot(rows, window):
    labels = []
    ys = []

    for method_key, method_title in TOP_METHODS_ONLY:
        result = compute_sep_score(rows, window, method_key)
        if result is not None:
            labels.append(method_title)
            ys.append(result["score"])

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(labels)), ys, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Top-patch aggregation")
    plt.ylabel("Separation score")
    plt.title(f"ID vs combined(RA21, Far) separation vs top patches\nwindow={window}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"separation_vs_top_method_w{window}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def gaussian_pdf(x, mu, sigma):
    if sigma <= 0:
        return 0.0
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def averaged_gaussian_pdf(x, mu_a, sigma_a, mu_b, sigma_b):
    return 0.5 * gaussian_pdf(x, mu_a, sigma_a) + 0.5 * gaussian_pdf(x, mu_b, sigma_b)


def find_intersection_with_average(
    mu_id, sigma_id,
    mu_ra, sigma_ra,
    mu_far, sigma_far,
    x_min, x_max,
    steps=20000
):
    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
    diffs = [
        gaussian_pdf(x, mu_id, sigma_id)
        - averaged_gaussian_pdf(x, mu_ra, sigma_ra, mu_far, sigma_far)
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


rows = read_rows(CSV_PATH)

# Plot mean-vs-window curves for every method
for method_key, method_title in METHODS:
    make_plot(rows, method_key, method_title)

print("Saved all mean plots.\n")

# Additional requested plots
make_separation_vs_window_plot(rows, "patch_top5", "Top 5 patch mean")
make_separation_vs_method_plot(rows, 96)
print("Saved requested separation plots.\n")

# ---- best setting finder: ID vs combined (RA21 + Far) mixture ----

best_combined = None

all_windows = sorted(set(r["window"] for r in rows))
all_methods = [m[0] for m in METHODS]

for window in all_windows:
    for method_key in all_methods:
        result = compute_sep_score(rows, window, method_key)
        if result is None:
            continue

        if best_combined is None or result["score"] > best_combined["score"]:
            best_combined = result

print("Best by ID vs combined(RA21, Far) separation:")
print(
    f"window={best_combined['window']}, "
    f"method={best_combined['method']}, "
    f"sep_score={best_combined['score']:.6f}, "
    f"raw_mean_distance={best_combined['raw_dist']:.6f}, "
    f"id_mean={best_combined['mu_id']:.6f}, "
    f"id_std={best_combined['std_id']:.6f}, "
    f"ra_mean={best_combined['mu_ra']:.6f}, "
    f"ra_std={best_combined['std_ra']:.6f}, "
    f"far_mean={best_combined['mu_far']:.6f}, "
    f"far_std={best_combined['std_far']:.6f}, "
    f"mix_mean={best_combined['mu_mix']:.6f}, "
    f"mix_std={best_combined['std_mix']:.6f}"
)

# ---- Gaussian plot for best setting + ID vs combined intersection ----

mu_id = best_combined["mu_id"]
std_id = best_combined["std_id"]
mu_ra = best_combined["mu_ra"]
std_ra = best_combined["std_ra"]
mu_far = best_combined["mu_far"]
std_far = best_combined["std_far"]

left = min(
    mu_id - 4 * std_id,
    mu_ra - 4 * std_ra,
    mu_far - 4 * std_far,
)
right = max(
    mu_id + 4 * std_id,
    mu_ra + 4 * std_ra,
    mu_far + 4 * std_far,
)

if left == right:
    left -= 1.0
    right += 1.0

thr_id_vs_combined = find_intersection_with_average(
    mu_id, std_id,
    mu_ra, std_ra,
    mu_far, std_far,
    left, right
)

num_points = 1000
xs = [left + (right - left) * i / (num_points - 1) for i in range(num_points)]

id_pdf = [gaussian_pdf(x, mu_id, std_id) for x in xs]
ra_pdf = [gaussian_pdf(x, mu_ra, std_ra) for x in xs]
far_pdf = [gaussian_pdf(x, mu_far, std_far) for x in xs]
mix_pdf = [averaged_gaussian_pdf(x, mu_ra, std_ra, mu_far, std_far) for x in xs]

# IEEE single-column width
fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.plot(xs, id_pdf, color="blue", label=f"ID")
ax.plot(xs, ra_pdf, color="red", label=f"RA21")
ax.plot(xs, far_pdf, color="green", label=f"Far OOD")
ax.plot(xs, mix_pdf, color="black", linestyle="--", label="Combined OOD")

if thr_id_vs_combined is not None:
    ax.axvline(
        thr_id_vs_combined,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold"
    )

ax.set_xlabel("Score")
ax.set_ylabel("PDF")
ax.grid(True, alpha=0.4)
ax.legend(loc="best", frameon=True)
fig.tight_layout()

out_path_png = (
    f"best_combined_gaussians_"
    f"{best_combined['method']}_w{best_combined['window']}_ieee.png"
)
out_path_pdf = (
    f"best_combined_gaussians_"
    f"{best_combined['method']}_w{best_combined['window']}_ieee.pdf"
)

fig.savefig(out_path_png, bbox_inches="tight")
fig.savefig(out_path_pdf, bbox_inches="tight")
plt.close(fig)

print(f"Saved Gaussian plot to {out_path_png}")
print(f"Saved Gaussian plot to {out_path_pdf}")
if thr_id_vs_combined is not None:
    print(f"ID vs combined(RA21, Far) intersection = {thr_id_vs_combined:.6f}")
else:
    print("No ID vs combined(RA21, Far) intersection found in plotting range.")