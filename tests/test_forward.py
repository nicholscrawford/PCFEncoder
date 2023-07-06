import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import numpy as np
from pcfencoder.pcfencoder import PCFEncoder

grid_size = [0.02, 0.06, 0.15, 0.375, 0.9375]
K_forward = [16, 16, 16, 16, 16]
K_propagate = [16, 16, 16, 16, 16]
K_self = [16, 16, 16, 16, 16]

npoint =1
coords = np.random.randn(npoint, 3)
norms = np.random.randn(npoint, 3)
color = np.random.randint(0, 255, (npoint, 3))

point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list \
    = PCFEncoder().subsample_and_knn(
        coord=coords, 
        norm=norms, 
        grid_size=grid_size, 
        K_self=K_self, 
        K_forward=K_forward, 
        K_propagate=K_propagate
        )

all_data = {}
all_data['point_list'] = point_list
all_data['nei_forward_list'] = nei_forward_list
all_data['nei_propagate_list'] = nei_propagate_list
all_data['nei_self_list'] = nei_self_list
all_data['surface_normal_list'] = norm_list
all_data['feature_list'] = [color.astype(np.float32)]

features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = PCFEncoder().collect_fn([all_data])

pred = PCFEncoder().pcf_backbone(
            features,
            pointclouds,
            edges_self,
            edges_forward,
            norms)

print(pred)
for it in pred:
    it = it.transpose(-1, -2)
    it = PCFEncoder().max_pool(it) + PCFEncoder().average_pool(it)
    it = it.transpose(-1, -2)
    print(it.shape)
    print(it)