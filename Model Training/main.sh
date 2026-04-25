wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 15 \
    --lr 5e-5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "SegFormer-b2-training-basic" \