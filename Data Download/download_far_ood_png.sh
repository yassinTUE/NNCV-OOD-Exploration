#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=3:00:00

set -e

mkdir -p data/far-ood
cd data/far-ood

base_url="http://r0k.us/graphics/kodak"
for i in $(seq -w 1 24); do
  wget -O "kodim${i}.png" "${base_url}/kodak/kodim${i}.png"
done

echo "Done. Downloaded Kodak PNG images into: $(pwd)"
find . -maxdepth 1 -type f -name '*.png' | sort
