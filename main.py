#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:45:04 2025

@author: rlc001
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_processing import prepare_data
from train import train_model
from train import make_snapshot_data
from evaluate import evaluate_model
import torch
from torch_geometric.loader import DataLoader

# === Prepare Data ===

work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"

node_features, target_vals, mask, edge_index = prepare_data(
    work_dir=work_dir,
    year_range=(1999, 2000),
    stations=["P22", "P23", "P24", "P25", "P26"],
    depths=[0.5, 10.5, 50.5, 100.5],
    variable="Temperature",
)

# === Split Data ===
T = node_features.shape[0]
# split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# split indices
train_end = int(train_ratio * T)
val_end = int((train_ratio + val_ratio) * T)

train_data = (node_features[:train_end],target_vals[:train_end],mask[:train_end], edge_index)
val_data = (node_features[train_end:val_end],target_vals[train_end:val_end],mask[train_end:val_end],edge_index)

# === Train Model ===
save_path = f"{work_dir}/GNN/best_model.pt"
model = train_model(train_data, val_data, n_epochs=200, save_path=save_path)

# === Evaluate Model ===
model.load_state_dict(torch.load(save_path))
test_data = make_snapshot_data(node_features[val_end:],target_vals[val_end:],mask[val_end:],edge_index)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

results = evaluate_model(
    model,
    test_loader,
    stations=['P19', 'P20', 'P21', 'P22'],
    depths=[2.5, 12.5, 27.5, 52.5, 102.5, 202.5, 302.5],
    plot_dir=f"{work_dir}/GNN/plots"
)
