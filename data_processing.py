#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:32:59 2025

@author: rlc001
"""


import pandas as pd
import numpy as np
import glob
import os
import xarray as xr
import json
from pathlib import Path


def load_ctd_data(ctd_data_file, start_year, end_year):
    """
    Load and process CTD csv files for a given year range.
    Returns an xarray.Dataset with dimensions (depth, station, time)).
    """

    df_all = pd.read_csv(ctd_data_file, comment="#")

    df_all["TIME"] = pd.to_datetime(df_all["TIME"], format="%Y-%m-%d %H:%M:%S")
    df_all = df_all.rename(
        columns={
            "LATITUDE": "Latitude",
            "LONGITUDE": "Longitude",
            "TEMPERATURE": "Temperature",
            "SALINITY": "Salinity",
            "OXYGEN_UMOL_KG": "Oxygen",
            "PRESSURE_BIN_CENTER": "Depth",
            "TIME": "time",
            "STATION_ID": "station"
        }
    )

    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year+1}-01-01")
    df_all = df_all[(df_all["time"] >= start_date) & (df_all["time"] < end_date)]

    # Sort and get unique coords
    depths = np.sort(df_all["Depth"].unique())
    stations = sorted(
        df_all["station"].unique(),
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    times = np.sort(df_all["time"].unique())

    # Build arrays
    variables = ["Temperature", "Salinity", "Oxygen", "Latitude", "Longitude"]
    data_dict = {var: np.full((len(times), len(stations), len(depths)), np.nan) for var in variables}

    for t_idx, t in enumerate(times):
        df_t = df_all[df_all["time"] == t]
        for s_idx, s in enumerate(stations):
            df_s = df_t[df_t["station"] == s]
            if df_s.empty:
                continue
            depth_idx = np.searchsorted(depths, df_s["Depth"])
            for var in variables:
                valid = (depth_idx >= 0) & (depth_idx < len(depths))
                data_dict[var][t_idx, s_idx, depth_idx[valid]] = df_s[var].values[valid]

    # Return as xarray dataset
    ds = xr.Dataset(
        {
            var: (("time", "station", "depth"), data_dict[var]) for var in variables
        },
        coords={
            "time": times,
            "station": stations,
            "depth": depths
        },
    )

    print(ds)

    ds["depth"].attrs["units"] = "m"
    ds["Temperature"].attrs["units"] = "deg C"
    ds["Salinity"].attrs["units"] = "PSU"
    ds["Oxygen"].attrs["units"] = "umol/kg"
    ds["Longitude"].attrs["units"] = "deg"
    ds["Latitude"].attrs["units"] = "deg"
        
    return ds


def normalize_dataset(ds, var_methods=None):
    """
    Normalize selected variables in an xarray.Dataset for ML.
    Returns:
      - normalized dataset
      - dictionary of scaling parameters for rescaling later
    """

    ds_norm = ds.copy(deep=True)
    scale_params = {}

    # Default normalization methods (can override with var_methods)
    default_methods = {
        "Temperature": "zscore",
        "Salinity": "minmax",
        "Oxygen": "zscore",
        "Bathymetry": "minmax",
        "Depth": "minmax",
        "Latitude": None,
        "Longitude": None,
    }

    if var_methods is None:
        var_methods = default_methods

    for var in ds.data_vars:
        method = var_methods.get(var, None)
        data = ds[var]

        if method == "zscore":
            mean_val = float(data.mean(skipna=True))
            std_val = float(data.std(skipna=True))
            ds_norm[var] = (data - mean_val) / std_val

            scale_params[var] = {
                "method": "zscore",
                "mean": mean_val,
                "std": std_val
            }

        elif method == "minmax":
            min_val = float(data.min(skipna=True))
            max_val = float(data.max(skipna=True))
            ds_norm[var] = (data - min_val) / (max_val - min_val)

            scale_params[var] = {
                "method": "minmax",
                "min": min_val,
                "max": max_val
            }

        else:
            # Variable not normalized (e.g., coordinates)
            scale_params[var] = {"method": None}
            continue

        print(f"Normalized {var} using {method}")

    return ds_norm, scale_params

def apply_normalization(ds, scale_params):
    """Apply precomputed normalization parameters to a dataset."""
    ds_norm = ds.copy(deep=True)
    for var, params in scale_params.items():
        if params["method"] == "zscore":
            mean_val = params["mean"]
            std_val = params["std"]
            ds_norm[var] = (ds[var] - mean_val) / std_val

        elif params["method"] == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            ds_norm[var] = (ds[var] - min_val) / (max_val - min_val)
        # else: leave unchanged
    return ds_norm


def make_synthetic_linep(time, stations, depths) -> xr.Dataset:
   
    T = len(time)
    D = len(depths)
    S = len(stations)
    rng = np.random.default_rng(0)
    data = np.zeros((T, S, D), dtype=np.float32)

    for ti, t in enumerate(time):
        seasonal = 4.0 * np.sin(2 * np.pi * (t.dt.month - 1) / 12.0)
        for si in range(S):
            for di, depth in enumerate(depths):
                val = seasonal
                val += 0.2 * si                         
                val += np.exp(-depth / 200.0)          
                val += 0.3 * np.sin(0.1 * si * ti / max(1, S))
                val += 0.5 * rng.normal()             
                data[ti, si, di] = val + 10

    ds = xr.Dataset({"Temperature": (("time", "station", "depth"), data)}, coords={"time": time, "station": stations, "depth": depths})

    return ds

def build_graph_structure(num_stations, num_depths):
    # node index increases down each depth column first, then moves to the next station.
    
    # Generates node edge connections
    edges = []
    for si in range(num_stations):
        for di in range(num_depths):
            node = si * num_depths + di
            # vertical neighbor (depth)
            if di < num_depths - 1:
                edges.append((node, si * num_depths + di + 1))
            # horizontal neighbor (station)
            if si < num_stations - 1:
                edges.append((node, (si+1) * num_depths + di))
    # make undirected
    edge_index = np.array(edges).T
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return edge_index

def reshape_to_graph_structure(ds_input: xr.DataArray, ds_target: xr.DataArray):
    """
    Reshape input and target datasets (time, station, depth)
    into (time, node, feature) format for GNNs.
    Returns node_features, target_vals, and mask.
    """
    num_times = ds_input.sizes["time"]
    num_stations = ds_input.sizes["station"]
    num_depths = ds_input.sizes["depth"]
    num_nodes = num_stations * num_depths
    num_features = len(ds_input.data_vars)
    
    ds_input_array = ds_input.to_array().values
    arr = np.moveaxis(ds_input_array, 1, 0)             # (time, features, stations, depths)
    arr = arr.transpose(0, 2, 3, 1)                     # (time, stations, depths, features)
    # Flatten => (time, node, features)
    input_vals_nodes = arr.reshape(num_times, num_nodes, num_features)
    
    ds_target_array = ds_target.to_array().values
    arr = np.moveaxis(ds_target_array, 1, 0)            # (time, 1, stations, depths)
    arr = arr.squeeze(axis=1)                           # (time, stations, depths)
    # Flatten => (time, node)
    target_vals_nodes = arr.reshape(num_times, num_nodes) 

    # Create mask and fill NaNs
    mask = ~np.isnan(target_vals_nodes)
    target_vals_nodes = np.nan_to_num(target_vals_nodes, nan=0.0)
     
    print("Input dimensions:")
    print(f"times: {num_times}, stations: {num_stations}, depths: {num_depths}, features: {num_features}")
    print(f"input features: {list(ds_input.data_vars)}")
    print(f"target feature: {list(ds_target.data_vars)}")
    
    print("Output dimensions")
    print(f"times: {num_times}, nodes: {num_nodes}, features: {num_features}")

    print("Input:", input_vals_nodes.shape)  # (time, nodes, features)
    print("Target:", target_vals_nodes.shape)  # (time, nodes)
    print("mask:", mask.shape)          # (time, nodes)

    return input_vals_nodes, target_vals_nodes, mask

def prepare_gnn_data(
    work_dir: Path,
    data_dir: Path,
    year_range: tuple[int, int],
    stations: list[str] | None = None,
    depths: list[float] | None = None,
    target_variable: str = "Temperature",
):
    
    ctd_filename = data_dir / "lineP_CTD_training.csv"

    #work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"
    #year_range = (1999, 2000)
    #target_variable = "Temperature"
    #stations = ["P22", "P23", "P24", "P25", "P26"]
    #depths = [0.5, 10.5, 50.5, 100.5]
    
    start_year, end_year = year_range
    
    # Load CTD observations (target)
    ds = load_ctd_data(ctd_filename, start_year, end_year)

    # Subset stations and depths
    #print(ds.station.values)
    if stations is not None: 
        ds = ds.sel(station=stations)
    # OR by exclusion
    #stations = ["P1", "P2", "P3"]
    #ds_subset = ds.drop_sel(station=stations)
    
    #print(ds.depth.values)
    if depths is not None: 
        ds = ds.sel(depth=depths)
    # OR by depth range
    # ds_mid = ds.sel(depth=slice(500, 2000)
    
    # Subset target variable
    ds_target = ds[[target_variable]]
    stations = ds_target['station']
    depths = ds_target['depth']
    
    # Generate synthetic line p temperature 'model' data
    # Replace this by loading model data
    ds_input = make_synthetic_linep(ds_target['time'], ds_target['station'], ds_target['depth'])
    
    # Add static variables
    # ****** FAKE bathymetry values**********
    bathymetry_in = xr.DataArray(
    [700, 720, 800, 850, 1000],
    dims=("station",),
    coords={"station": ds_input.station},
    name="Bathymetry"
    )
    ds_input["Bathymetry"] = bathymetry_in.broadcast_like(ds_input[target_variable])
    
    depth_in = xr.DataArray(
    depths,
    dims=("depth",),
    coords={"depth": ds_input.depth},
    name="Depth"
    )
    ds_input["Depth"] = depth_in.broadcast_like(ds_input[target_variable])
    
    # === Split Data into train, validation, test ===
    T = ds_input.sizes["time"]
    # split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # split indices
    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)
    
    ds_input_train = ds_input.isel(time=slice(0, train_end))
    ds_input_val   = ds_input.isel(time=slice(train_end, val_end))
    ds_input_test  = ds_input.isel(time=slice(val_end, T))
    
    ds_target_train = ds_target.isel(time=slice(0, train_end))
    ds_target_val   = ds_target.isel(time=slice(train_end, val_end))
    ds_target_test  = ds_target.isel(time=slice(val_end, T))

    # Normalization
    # Compute scale parameters from training data and apply to validation and test
    ds_input_train_norm, scale_params_in = normalize_dataset(ds_input_train)
    # Save input normalization parameters
    with open(f"{work_dir}/graph-neural-net/scale_params_in.json", "w") as f:
        json.dump(scale_params_in, f, indent=2)
    
    # Apply same normalization to validation & test inputs
    ds_input_val_norm  = apply_normalization(ds_input_val, scale_params_in)
    ds_input_test_norm = apply_normalization(ds_input_test, scale_params_in)
    
    ds_target_train_norm, scale_params_target = normalize_dataset(ds_target_train)
    # Save target normalization parameters
    with open(f"{work_dir}/graph-neural-net/scale_params_target.json", "w") as f:
        json.dump(scale_params_target, f, indent=2)
    
    # Apply same normalization to validation & test targets
    ds_target_val_norm  = apply_normalization(ds_target_val, scale_params_target)
    ds_target_test_norm = apply_normalization(ds_target_test, scale_params_target)

    # reshape data into graph structure, and compute target value mask
    print("\nTraining:")
    node_features_train, target_vals_train, mask_train = reshape_to_graph_structure(ds_input_train_norm, ds_target_train_norm)
    print("\nValidation:")
    node_features_val, target_vals_val, mask_val = reshape_to_graph_structure(ds_input_val_norm, ds_target_val_norm)
    print("\nTesting:")
    node_features_test, target_vals_test, mask_test = reshape_to_graph_structure(ds_input_test_norm, ds_target_test_norm)

    # generate edge connections from graph nodes
    edge_index = build_graph_structure(len(ds_target['station']), len(ds_target['depth']))
    
    train_data = (node_features_train, target_vals_train, mask_train, edge_index)
    val_data = (node_features_val, target_vals_val, mask_val, edge_index)
    test_data = (node_features_test, target_vals_test, mask_test, edge_index)
    
    return train_data, val_data, test_data, stations, depths 

