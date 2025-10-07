#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:35:48 2025

@author: rlc001
"""

import pandas as pd
import numpy as np
import xarray as xr


def load_station_variable(InDir, stations, variable):
    
    station_datasets = []
    
    for station in stations:
        
        csv_file = f"{InDir}/Station{station}_5mInterp{variable}_4amber.csv"
        df = pd.read_csv(csv_file)
        
        depth = df.iloc[:, 0].values
        time_cols = df.columns[1:]
        times = pd.to_datetime(time_cols, format="%m/%Y")
        
        values = df.iloc[:, 1:].to_numpy()
        
        da = xr.DataArray(
            values,
            dims=("depth", "time"),
            coords={"depth": depth, "time": times},
            name=variable
        )
        
        da = da.expand_dims(station=[station])
        station_datasets.append(da)
    
    ds = xr.concat(station_datasets, dim="station")
    
    return ds.to_dataset(name=variable)


def make_synthetic_linep(start_date,end_date,depths,stations) -> xr.Dataset:
   
    rng = np.random.default_rng(0)

    time = pd.date_range(start=start_date, end=end_date, freq="MS")
    T = len(time)
    D = len(depths)
    num_stations = len(stations)

    data = np.zeros((num_stations, D, T), dtype=np.float32)

    for ti, t in enumerate(time):
        seasonal = 4.0 * np.sin(2 * np.pi * (t.month - 1) / 12.0)
        for si in range(num_stations):
            for di, depth in enumerate(depths):
                val = seasonal
                val += 0.2 * si                         
                val += np.exp(-depth / 200.0)          
                val += 0.3 * np.sin(0.1 * si * ti / max(1, num_stations))
                val += 0.5 * rng.normal()             
                data[si, di, ti] = val + 10

    ds = xr.Dataset({"Temp": (("station", "depth", "time"), data)}, coords={"station": stations, "depth": depths,"time": time})

    return ds


def build_edge_index(num_stations, num_depths):
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

def prepare_linep_data(ds_mod, ds_obs, variable):
    stations = ds_mod.station.values
    depths = ds_mod.depth.values
    times = ds_mod.time.values

    num_stations = len(stations)
    num_depths = len(depths)
    num_nodes = num_stations * num_depths
    num_times = len(times)

    # Flatten (station, depth) -> node index
    def node_index(sta_i, dep_i):
        return sta_i * num_depths + dep_i

    mod_vals = ds_mod[variable].values  # shape (station, depth, time)
    mod_vals = np.transpose(mod_vals, (2,0,1))  # -> (time, station, depth)
    mod_vals = mod_vals.reshape(num_times, num_nodes, 1)  # (time, node, feat=1)

    obs_vals = ds_obs[variable].values  # shape (station, depth, time)
    obs_vals = np.transpose(obs_vals, (2,0,1))  # (time, station, depth)
    obs_vals = obs_vals.reshape(num_times, num_nodes)  # (time, node)
    
    # normalize
    mean_X = np.mean(mod_vals)
    std_X = np.std(mod_vals)
    
    model_norm = (mod_vals - mean_X) / std_X
    obs_norm = (obs_vals - mean_X) / std_X
    
    # Mask where obs is not NaN
    mask = ~np.isnan(obs_vals)

    # Replace NaN in obs with 0.0 for storage
    #obs_vals = np.nan_to_num(obs_vals, nan=0.0)

    return model_norm, obs_norm, mask, stations, depths, times

def prepare_data(InDir, variable, stations, depths, start_date, end_date):
    
    #InDir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"
    #stations = ['P19', 'P20', 'P21', 'P22']
    #variable = 'Temp'
    #start_date = "2015-01-01"
    #end_date = "2024-12-31"
    #depths=[2.5, 12.5, 27.5, 52.5, 102.5, 202.5, 302.5]
    
    # load observations for stations and variable
    ds_obs = load_station_variable(InDir, stations, variable=variable)
    
    # subset data in time and depths
    ds_obs = ds_obs.sel(time=slice(start_date, end_date))
    ds_obs = ds_obs.sel(depth=depths)
    
    # generate synthetic line p 'model' data 
    ds_mod = make_synthetic_linep(start_date, end_date, depths, stations)
    
    # generate edge connections from graph nodes
    edge_index = build_edge_index(len(stations), len(depths))
    
    # fill missing observation time steps, reindex the times such that model and obs time match
    all_times = xr.concat([ds_mod.time, ds_obs.time], dim="time").to_index().unique().sort_values()
    ds_mod = ds_mod.reindex(time=all_times)
    ds_obs = ds_obs.reindex(time=all_times)
    
    # prepare data reshape and normalize
    model_vals, target_vals, mask, stations, depths, times = prepare_linep_data(ds_mod, ds_obs, variable)
    
    # add static features ex depth, bathymetry, lon, lat
    # I've just added a 'fake' bathymetry depth for now
    depth_vals = np.tile(depths, len(stations))
    bathy_vals = np.repeat(np.array([700, 720, 800, 850]), len(depths))
    #lon_vals = 
    #lat_vals = 
    
    static_feats = np.stack([bathy_vals, depth_vals], axis=1)  # (N, num_stat_feats)
    
    T = len(all_times)
    node_static_feats = np.tile(static_feats[None, :, :], (T, 1, 1))
    
    # combine model values with static features
    node_features = np.concatenate([model_vals, node_static_feats], axis=2)
    
    print("Stations:", stations)  # (time, nodes, features)
    print(f"S: {len(stations)}, D: {len(depths)}, N: {len(stations)*len(depths)}, T: {len(times)}, F: {node_features.shape[2]}")
    print("model_vals:", model_vals.shape)  # (time, nodes, features)
    print("target_vals:", target_vals.shape)  # (time, nodes)
    print("node_features:", node_features.shape)  # (time, nodes, features)
    print("mask:", mask.shape)          # (time, nodes)
    
    return node_features, target_vals, mask, edge_index