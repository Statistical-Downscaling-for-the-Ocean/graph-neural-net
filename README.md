# graph-neural-net
This repository contains the code related to a graph neural net approach to filling in Line-P data.


Work flow for a GNN from main.py.

The general idea is to build a graph with nodes connected by adjacent depths and stations and to use the modeled output variables to predict observations.

## Notes
* The code currently uses a single variable (temp) and static variables (ex. bathymetry, lat, lon ...) to predict a single target variable (obs temp).
* The code currently selects a subset of the stations and selects specific depths (constant for all stations) to build the graph. For a transductive GNN the graph must be constant in time which raises a challenge with respect to how to define the graph if the depth of observations is not consistent at each station and in time. Some sort of interpolation may be needed.


## Data processing
In data_processing.py I load the interpolated csv data (ex. StationP19_5mInterpTemp_4amber.csv). I know we shouldn't be using the interpolated but the other dataset has some issues like missing time data and also it was simpler to have all the observations at the same depths for all stations for now. I also generate a synthetic lineP data that is in place of the model data.
Also in data_processing.py I build the graph connections in build_edge_index. This function is quite simple for now. Given the number of stations and depths it connect the graph through adjacent depths and stations. This is easy with a 'rectangular' graph but will become more challenging when there are a different number of depths measurements for different stations. Just a note, essentially the 'grid' is collapsed down to a list of nodes with indexes for training and reshaped back into (stations, depth) for evaluation but if the graph becomes unrectangular this will have to be taken into account as well. 

* I have normalized the data, but I have not stored the mean and std values to correct the model output for evaluation.

## Model details
For now there is no time dependency in the model. The model is defined in model_definitions.py as SnapshotGNN.

This SnapshotGNN is a node-level regression model. It takes per-node input features (ex. temp, depth, neighboring temperature) and predicts a scalar per node (ex. observed temperature) for that same snapshot. Each snapshot (time step) is treated as one static graph, with nodes connected via edge_index. You train the model across many snapshots (so time steps act as separate samples).

Architecture:
You can flexibly choose between three GNN types:
* SAGEConv: GraphSAGE (samples & aggregates neighbors - good for inductive learning)
* GCNConv: classical Graph Convolutional Network (smooths over neighbors)
* GraphConv: simple averaging variant
Default is 'sage', which is great for physical data because it generalizes well.

The model stacks multiple convolution layers (default n_layers=2). The first layer maps your input features (ex. [temp, depth, lat, lon]) to a hidden space. Intermediate layers propagate and mix node information via graph edges i.e., each node learns from its connected neighbors. The learned hidden representation is converted into the target value (out_channels=1).

## Training
The model training is done in train.py. It's pretty standard I think. I use MSE as training criterion. The loss is only computed where there are valid observations.

* Could include early stopping.

## Evaluation 
evalute_model in evaluate.py generates predictions from test data and compares them to observations and creates some plots. 
