import torch
import torch.nn as nn

import ml_pointconvformer as pcf

class PCFEncoder(nn.Module):
    def __init_(self):
        self.pcf_backbone = 0

    def forward(self, x):
        return x