# Graph Neural Network for Statistical Ocean Downscaling

This repository contains a Graph Neural Network (GNN) implementation for statistical downscaling of ocean observations, specifically designed for filling gaps in Line-P oceanographic data using CTD measurements and model predictions.

## Overview

The project implements a transductive Graph Neural Network approach that treats oceanographic stations and depth levels as nodes in a graph, with edges connecting spatially adjacent locations. The model learns to predict observed oceanographic variables (e.g., temperature) from model outputs and static environmental features.

### Key Features

- **Graph-based spatial modeling**: Nodes represent station-depth combinations with edges connecting adjacent stations and depths
- **Subsetting**: Select a subset of the stations and depths for testing and builds the graph connections accordingly 
- **Multi-variable input support**: Handles temperature, salinity, oxygen, bathymetry, and depth as input features
- **Flexible architecture**: Supports GraphSAGE, GCN, and GraphConv layers
- **Comprehensive evaluation**: Includes RMSE, MAE, R², and bias metrics with visualization
- **Proper data normalization**: Z-score and min-max scaling with parameter persistence
- **Train/validation/test splits**: Temporal splitting with proper normalization workflow

## Installation

### Using the provided setup script:
./setup_env.sh

## Workflow

### Data processing (`data_processing.py`)

`load_ctd_data()`: Loads CTD observations from CSV files (LineP_ctds_YYYY_binned_1m.csv)
`normalize_dataset()`: Applies z-score or min-max normalization with scaling parameters computed from only the training set (`scale_params.json` files are saved with the scaling parameters to denormalize later)
`reshape_to_graph_structure()`: Reshapes the data to appropriate graph structure
`build_graph_structure()`: Creates edge connectivity matrix connecting the graph through adjacent depths and stations
`prepare_gnn_data()`: Complete preprocessing pipeline: loading data, normalization, splitting into training, validation and test, reshaping, building graph

#### Graph Structure
Nodes: Each station-depth combination becomes a graph node
Edges: Connect adjacent stations horizontally and adjacent depths vertically
Features: Each node has multiple features (temperature, depth, bathymetry, etc.)
Target: Single scalar value per node (e.g., observed temperature)

### Model Architecture (`model_definitions.py`)

`SnapshotGNN` is a node-level regression model. It takes per-node input features (ex. temp, depth ... ) and predicts a scalar per node (ex. observed temperature) for that same snapshot. Each snapshot (time step) is treated as one static graph, with nodes connected via edge_index. The model is trained across many snapshots (so time steps act as separate samples).

Architecture:
You can flexibly choose between three GNN types:
* SAGEConv: GraphSAGE (samples & aggregates neighbors - good for inductive learning)
* GCNConv: classical Graph Convolutional Network (smooths over neighbors)
* GraphConv: simple averaging variant
Default is 'sage', which is great for physical data because it generalizes well.

The model stacks multiple convolution layers (default n_layers=2). The first layer maps your input features (ex. [temp, depth, ...]) to a hidden space. Intermediate layers propagate and mix node information via graph edges i.e., each node learns from its connected neighbors. The learned hidden representation is converted into the target value (out_channels=1).

### Training (`train.py`)

Temporal snapshots: Each time step treated as independent graph
Masked loss: Only computed on nodes with valid observations (MSE as criterion)
Adam optimizer with weight decay regularization
Model checkpointing saves best weights

### Evaluation (`evaluate.py`)

Denormalization: Converts predictions back to physical units
Computes metrics: RMSE, MAE, R², bias
Visualization:
  Scatter plot of predictions vs observations
  Residual histogram
  Spatial heatmap of mean residuals by station/depth

## File structure

graph-neural-net/
- main.py                  # Main execution script
- data_processing.py       # Data loading and preprocessing
- model_definitions.py     # GNN model architectures  
- train.py                 # Training utilities
- evaluate.py              # Evaluation and visualization
- requirements.txt         # Python dependencies
- setup_env.sh             # Environment setup script
- scale_params_in.json     # Input normalization parameters  
- scale_params_target.json # Target normalization parameters
- best_model.pt            # Saved model weights
- plots/                   # Generated evaluation plots
    - scatter_pred_vs_obs.png
    - residual_hist.png
    - residual_heatmap.png

## Next steps

### Must dos
 - Replace synthetic model data with real ocean model output
 - Get actual bathymetry values

### Current limitations
 - No temporal dependencies (treats each time step independently)
 - Limited to single target variable prediction
 - Static graph structure (same stations/depths for all time steps)

### Potential improvements
 - Temporal GNN: Add recurrent or attention mechanisms for time dependencies
 - Multi-target prediction: Simultaneously predict multiple variables
 - Inductive learning: Allow prediction at new station/depth locations
