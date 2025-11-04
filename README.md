# graph-neural-net
This repository contains the code related to a graph neural net approach to filling in Line-P data.


Work flow for a GNN from main.py.

The general idea is to build a graph with nodes connected by adjacent depths and stations and to use the model output to predict observations.

## Notes
* The code currently uses a single variable (temperature) and static variables (ex. bathymetry, depth ...) to predict a single target variable (observe temperature from ctds).
* You can select a subset of the stations and depths for testing and builds the graph connections accordingly. For a transductive GNN the graph must be constant in time.

## Data processing
In main.py the `prepare_gnn_data` function
- Loads the target observations (Line P ctd observations, files LineP_ctds_YYYY_binned_1m.csv, function `load_ctd_data`).
- Loads the model data predictors (For now synthetic lineP data is generated in place of real model data).
- Splits the the data into training, validation and testings sets.
- Normalizes all sets of data with scaling parameters computed from only the training set (`scale_params.json` files are saved with the scaling parameters to denormalize later).
- Reshapes the data to appropriate graph structure (`reshape_to_graph_structure`).
- Builds graph connectivity (`build_graph_structure` takes number of stations and depths and connect the graph through adjacent depths and stations).


## Model details
For now there is no time dependency in the model. The model is defined in `model_definitions.py` as `SnapshotGNN`.

This SnapshotGNN is a node-level regression model. It takes per-node input features (ex. temp, depth ... ) and predicts a scalar per node (ex. observed temperature) for that same snapshot. Each snapshot (time step) is treated as one static graph, with nodes connected via edge_index. The model is trained across many snapshots (so time steps act as separate samples).

Architecture:
You can flexibly choose between three GNN types:
* SAGEConv: GraphSAGE (samples & aggregates neighbors - good for inductive learning)
* GCNConv: classical Graph Convolutional Network (smooths over neighbors)
* GraphConv: simple averaging variant
Default is 'sage', which is great for physical data because it generalizes well.

The model stacks multiple convolution layers (default n_layers=2). The first layer maps your input features (ex. [temp, depth, ...]) to a hidden space. Intermediate layers propagate and mix node information via graph edges i.e., each node learns from its connected neighbors. The learned hidden representation is converted into the target value (out_channels=1).

## Training
The model training is done in train.py with MSE as training criterion. The loss is only computed where there are valid observations.

## Evaluation 
`evalute_model` in `evaluate.py` generates predictions from the testing data and compares them to valid observations and creates some plots. 
