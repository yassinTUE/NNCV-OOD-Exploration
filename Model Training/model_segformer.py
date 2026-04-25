import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class Model(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            num_labels=n_classes,
            ignore_mismatched_sizes=False,
        )

    def forward(self, x):
        out = self.model(pixel_values=x)
        return out.logits