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


def load_ctd_data(work_dir, start_year, end_year):
    """
    Load and process CTD csv files for a given year range.
    Returns an xarray.Dataset with dimensions (depth, station, time)).
    """
    
    data_dir = f"{work_dir}/data/lineP_ctds"
    
    # Collect files by year
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    year_files = [
        f for f in all_files
        if any(str(y) in os.path.basename(f) for y in range(start_year, end_year + 1))
    ]

    if not year_files:
        raise FileNotFoundError(f"No csv files found for years {start_year}-{end_year} in {data_dir}")

    # Load and concatenate all data
    df_list = []
    for file in year_files:
        print(f"Loading {os.path.basename(file)} ...")
        df = pd.read_csv(file)
        df = df.rename(columns={
            "latitude": "Latitude",
            "longitude": "Longitude",
            "CTDTMP_ITS90_DEG_C": "Temperature",
            "SALINITY_PSS78": "Salinity",
            "OXYGEN_UMOL_KG": "Oxygen",
            "PRS_bin_cntr": "Depth",
        })
        df["time"] = pd.to_datetime(df["time"])
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)

    # Sort and get unique coords
    depths = np.sort(df_all["Depth"].unique())
    stations = sorted(
        df_all["closest_linep_station_name"].unique(),
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    times = np.sort(df_all["time"].unique())

    # Build arrays
    variables = ["Temperature", "Salinity", "Oxygen", "Latitude", "Longitude"]
    data_dict = {var: np.full((len(times), len(stations), len(depths)), np.nan) for var in variables}

    for t_idx, t in enumerate(times):
        df_t = df_all[df_all["time"] == t]
        for s_idx, s in enumerate(stations):
            df_s = df_t[df_t["closest_linep_station_name"] == s]
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
    ds["depth"].attrs["units"] = "m"
    ds["Temperature"].attrs["units"] = "deg C"
    ds["Salinity"].attrs["units"] = "PSS-78"
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

def denormalize_variable(var_name, data_norm, scale_params):
    """Rescale normalized data back to original physical units."""
    params = scale_params[var_name]
    method = params["method"]

    if method == "zscore":
        return data_norm * params["std"] + params["mean"]
    elif method == "minmax":
        return data_norm * (params["max"] - params["min"]) + params["min"]
    else:
        return data_norm  # unchanged


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

def build_edge_index(num_stations, num_depths):
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

def prepare_graph_data(ds_input: xr.DataArray, ds_target: xr.DataArray):
    """
    Reshape input and target datasets (time, station, depth)
    into (time, node, feature) format for GNNs.
    Returns node_features, target_vals, and mask.
    """
    num_times = ds_input.sizes["time"]
    num_stations = ds_input.sizes["station"]
    num_depths = ds_input.sizes["depth"]
    num_nodes = num_stations * num_depths

    # Flatten (time, station, depth) → (time, node)
    input_vals_nodes = ds_input.values.reshape(num_times, num_nodes)
    target_vals_nodes = ds_target.values.reshape(num_times, num_nodes)

    # Create mask and fill NaNs
    mask = ~np.isnan(target_vals_nodes)
    target_vals_nodes = np.nan_to_num(target_vals_nodes, nan=0.0)

    # Static features
    depths = ds_target.depth.values
    depth_nodes = np.tile(depths, num_stations)
    # ****** FAKE bathymetry values**********
    bathy_nodes = np.repeat(np.array([700, 720, 800, 850, 1000]), num_depths)

    static_feats = np.stack([bathy_nodes, depth_nodes], axis=1)

    node_static_feats = np.tile(static_feats[None, :, :], (num_times, 1, 1))
    node_features = np.concatenate(
        [input_vals_nodes[..., np.newaxis], node_static_feats], axis=-1
    )
    
    print(f"S: {num_stations}, D: {num_depths}, N: {num_nodes}, T: {num_times}, F: {node_features.shape[2]}")
    print("Input:", input_vals_nodes.shape)  # (time, nodes, features)
    print("Target:", target_vals_nodes.shape)  # (time, nodes)
    print("node_features:", node_features.shape)  # (time, nodes, features)
    print("mask:", mask.shape)          # (time, nodes)

    return node_features, target_vals_nodes, mask


#%%

def prepare_data(
    work_dir: str,
    year_range: tuple[int, int],
    stations: list[str] | None = None,
    depths: list[float] | None = None,
    variable: str = "Temperature",
):
    
    #work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"
    #year_range = (1999, 2000)
    #variable = "Temperature"
    #stations = ["P22", "P23", "P24", "P25", "P26"]
    #depths = [0.5, 10.5, 50.5, 100.5]
    
    start_year, end_year = year_range
    ds = load_ctd_data(work_dir, start_year, end_year)
    
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
    # ds_mid = ds.sel(depth=slice(500, 2000))
    
    # Normalize dataset
    ds, scale_params = normalize_dataset(ds)
    # Example denormalization for model output
    #temp_original = denormalize_variable("Temperature", model_output_temp, scale_params)
    
    # Subset variables
    ds_target = ds[variable]
    
    # Generate synthetic line p temperature 'model' data 
    ds_input = make_synthetic_linep(ds_target['time'], ds_target['station'], ds_target['depth'])
    ds_input = ds_input[variable]
    
    # generate edge connections from graph nodes
    edge_index = build_edge_index(len(ds_target['station']), len(ds_target['depth']))
    
    # reshape data into graph structure
    node_features, target_vals_nodes, mask = prepare_graph_data(ds_input, ds_target)
    
    return node_features, target_vals_nodes, mask, edge_index

