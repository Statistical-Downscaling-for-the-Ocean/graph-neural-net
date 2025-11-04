#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:45:04 2025

@author: rlc001
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_processing import prepare_gnn_data
from train import train_model
from train import make_snapshot_data
from evaluate import evaluate_model
import torch
from torch_geometric.loader import DataLoader

# === Prepare Data ===
work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"

train_data, val_data, test_data, stations, depths = prepare_gnn_data(
    work_dir=work_dir,
    year_range=(1999, 2000),
    stations=["P22", "P23", "P24", "P25", "P26"],
    depths=[0.5, 10.5, 50.5, 100.5],
    target_variable="Temperature",
)

# === Train Model ===
save_path = f"{work_dir}/graph-neural-net/best_model.pt"
model = train_model(train_data, val_data, n_epochs=100, save_path=save_path)

# === Evaluate Model ===
model.load_state_dict(torch.load(save_path))
node_features_test, target_vals_test, mask_test, edge_index = test_data
test_data = make_snapshot_data(node_features_test, target_vals_test, mask_test, edge_index)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

results = evaluate_model(
    model,
    test_loader,
    target_variable="Temperature",
    stations=stations.values,
    depths=depths.values,
    work_dir=work_dir
)
