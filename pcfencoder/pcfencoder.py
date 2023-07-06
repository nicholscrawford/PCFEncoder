import torch
import torch.nn as nn

from pcfencoder.ml_pointconvformer.model_architecture import PCF_Normal
from pcfencoder.ml_pointconvformer.datasetCommon import subsample_and_knn, collect_fn

class PCFEncoder(nn.Module):
    def __init__(self,
                input_grid_size = 0.02, #2cm
                output_embed_size = 100
                ):
        super(PCFEncoder, self).__init__()
        self.collect_fn = collect_fn
        self.subsample_and_knn = subsample_and_knn #https://github.com/apple/ml-pointconvformer/blob/main/scannet_data_loader_color_DDP.py#L222
        self.pcf_backbone, self.conf = PCF_Normal(input_grid_size=input_grid_size)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_embed_size)
        self.average_pool = torch.nn.AdaptiveAvgPool1d(output_embed_size)

    def forward(self, x):
        x = self.subsample_and_knn(x)
        x = self.pcf_backbone(x)
        x = self.max_pool(x) + self.average_pool(x)
        return x