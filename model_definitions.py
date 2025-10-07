#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:19:21 2025

@author: rlc001
"""

import torch
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv

class SnapshotGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=1, n_layers=2, conv_type='sage'):
        super().__init__()
        conv = {'sage': SAGEConv, 'gcn': GCNConv, 'graph': GraphConv}[conv_type]
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv(in_channels, hidden))
        for _ in range(n_layers-1):
            self.convs.append(conv(hidden, hidden))
        self.lin = torch.nn.Linear(hidden, out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        for c in self.convs:
            x = c(x, edge_index)
            x = self.act(x)
        return self.lin(x).squeeze(-1)  # shape: [num_nodes]


