#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="$PWD"
OUT_DIR="$ROOT_DIR/data/near-ood-roadanomaly21"
WORK_DIR="$OUT_DIR/_work"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Downloading RoadAnomaly21 from SegmentMeIfYouCan (Zenodo mirror)..."
wget -O roadanomaly21.zip \
  "https://zenodo.org/record/5270237/files/dataset_AnomalyTrack.zip?download=1"

echo "Extracting archive..."
unzip -q roadanomaly21.zip -d extracted

echo "Collecting RGB PNG images..."
mkdir -p "$OUT_DIR"
find extracted -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | while read -r f; do
  base="$(basename "$f")"
  lower="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"

  case "$lower" in
    *label*|*labels*|*mask*|*gt*|*anno*|*annotation*)
      ;;
    *)
      cp "$f" "$OUT_DIR/$base"
      ;;
  esac
done

rm -rf "$WORK_DIR"

echo "Done. Images saved to: $OUT_DIR"
echo "Image count:"
find "$OUT_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l
