#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:44:02 2025

@author: rlc001
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from model_definitions import SnapshotGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_snapshot_data(node_features, target_vals, mask, edge_index):
    T, N, F = node_features.shape
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data_list = []
    for t in range(T):
        data = Data(
            x=torch.tensor(node_features[t], dtype=torch.float32),
            y=torch.tensor(target_vals[t], dtype=torch.float32),
            edge_index=edge_index,
            mask=torch.tensor(mask[t], dtype=torch.bool)
        )
        data_list.append(data)
    return data_list

def evaluate_snapshot(model, loader):
    model.eval()
    ys, ys_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            mask = data.mask
            if mask.sum() == 0:
                continue
            ys.extend(data.y[mask].cpu().numpy())
            ys_pred.extend(out[mask].cpu().numpy())
    if len(ys) == 0:
        return np.nan
    mse = mean_squared_error(ys, ys_pred)
    return mse

def train_snapshot_model(model, train_loader, val_loader=None, lr=1e-3, wd=1e-5, epochs=200, save_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_mse = np.inf
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            mask = data.mask
            if mask.sum() == 0:
                continue
            loss = F.mse_loss(out[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # validation
        if val_loader is not None:
            val_mse = evaluate_snapshot(model, val_loader)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = model.state_dict()
                if save_path is not None:
                    torch.save(best_state, save_path)
                    print(f"Saved best model (val_mse={best_val_mse:.4f})")
                
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | TrainLoss={total_loss:.4f} | ValMSE={val_mse if val_loader else 'N/A'}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_model(data_train, data_val, n_epochs=200, save_path=None):
    node_features, target_vals, mask, edge_index = data_train
    val_node_features, val_target_vals, val_mask, _ = data_val

    train_data = make_snapshot_data(node_features, target_vals, mask, edge_index)
    val_data = make_snapshot_data(val_node_features, val_target_vals, val_mask, edge_index)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    model = SnapshotGNN(in_channels=node_features.shape[-1], hidden=64, n_layers=2)
    model = train_snapshot_model(model, train_loader, val_loader, epochs=n_epochs, save_path=save_path)
    
    return model
