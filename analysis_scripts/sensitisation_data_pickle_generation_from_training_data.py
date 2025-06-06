#!/usr/bin/env python3
"""
CHR2 Sensitisation Data Pickle Generation Script
Pattern Learning Paper - Supplementary Figure 6
Author: Anzal (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/

This script extracts and processes training data to generate pickle files 
for CHR2 sensitisation analysis (violin plots of spike amplitudes, 
TTL rising edges, and LFP minimas over time).

This script replicates exactly the logic from the prototyping.ipynb notebook.
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from scipy.signal import find_peaks
import sys

def subtract_baseline(trace, sampling_rate, bl_period_in_ms):
    """
    Subtract baseline from trace using the baseline period - matches the notebook exactly.
    """
    bl_period = bl_period_in_ms / 1000
    bl_duration = int(sampling_rate * bl_period)
    bl = np.mean(trace[:bl_duration])
    bl_trace = trace - bl
    return bl_trace

def detect_ttl_rising_edges(ttl_trace, threshold=2.5):
    """
    Detect TTL rising edges in the trace - matches the notebook logic.
    """
    above_threshold = ttl_trace > threshold
    rising_edges = []
    
    for i in range(1, len(above_threshold)):
        if above_threshold[i] and not above_threshold[i-1]:
            rising_edges.append(i)
    
    return rising_edges

def detect_first_spike(trace, ttl_rising_edge, window_ms=50, sampling_rate=10000):
    """
    Detect the first spike after TTL rising edge - matches the notebook exactly.
    """
    window_samples = int(window_ms * sampling_rate / 1000)
    end_idx = min(ttl_rising_edge + window_samples, len(trace))
    
    if ttl_rising_edge >= len(trace):
        return None, None
    
    trace_section = trace[ttl_rising_edge:end_idx]
    
    if len(trace_section) > 0:
        # Find the maximum value in the window (spike detection)
        max_idx = np.argmax(trace_section)
        spike_idx = ttl_rising_edge + max_idx
        spike_val = trace_section[max_idx]
        
        # Only return if above a threshold (matches notebook exactly)
        if spike_val > 40:  # 40mV threshold as in notebook
            return spike_idx, spike_val
    
    return None, None

def detect_most_minimal(trace, ttl_rising_edge, window_ms=50, sampling_rate=10000):
    """
    Detect the most minimal value after TTL rising edge - matches the notebook exactly.
    """
    window_samples = int(window_ms * sampling_rate / 1000)
    end_idx = min(ttl_rising_edge + window_samples, len(trace))
    
    if ttl_rising_edge >= len(trace):
        return None, None
    
    trace_section = trace[ttl_rising_edge:end_idx]
    
    if len(trace_section) > 0:
        min_idx = np.argmin(trace_section)
        minima_val = trace_section[min_idx]
        return ttl_rising_edge + min_idx, minima_val
    
    return None, None

def save_plot_data_to_pickle(cell_data, filename="sensitisation_plot_data.pkl"):
    """
    Saves the required plot data into a pickle file - exactly as in the notebook.
    Only saves data where "frame_id" contains "training". Baseline subtraction is applied to all traces.
    
    Parameters:
    cell_data (DataFrame): The input data containing the traces.
    filename (str): The name of the pickle file where the data will be saved.
    """
    plot_data = []
    
    # Filter to include only rows where "frame_id" contains "training"
    cell_data = cell_data[cell_data["frame_id"].str.contains("training", case=False, na=False)]
    
    if cell_data.empty:
        print("No training data found in the dataset.")
        return
    
    print(f"Processing {len(cell_data['cell_ID'].unique())} cells for training data...")
    
    # Group by cell_ID and trial_no, preserving important information
    cell_grp = cell_data.groupby(by="cell_ID")
    
    for cell, cell_data_group in cell_grp:
        pre_post_status = cell_data_group["pre_post_status"].unique()[0]  # Assuming one status per cell
        sampling_rate = int(cell_data_group["sampling_rate(Hz)"].unique()[0])
        trial_grp = cell_data_group.groupby(by="trial_no")
        
        for trial, trial_data in trial_grp:
            # Extract trace data and apply baseline subtraction (exactly as in notebook)
            trace = trial_data["cell_trace(mV)"].to_numpy()
            trace = subtract_baseline(trace, sampling_rate, 5)  # 5ms baseline
            
            field_trace = trial_data["field_trace(mV)"].to_numpy()
            field_trace = subtract_baseline(field_trace, sampling_rate, 2)  # 2ms baseline
            
            ttl_trace = trial_data["ttl_trace(V)"].to_numpy()
            ttl_trace = subtract_baseline(ttl_trace, sampling_rate, 5)  # 5ms baseline

            # Detect TTL rising edges using a more sensitive threshold
            ttl_rising_edges = detect_ttl_rising_edges(ttl_trace, threshold=0.5)

            # Detect spikes and minima (after baseline subtraction)
            spikes = []
            minimas = []
            for edge in ttl_rising_edges:
                spike_idx, spike_val = detect_first_spike(trace, edge, sampling_rate=sampling_rate)
                minima_idx, minima_val = detect_most_minimal(field_trace, edge, sampling_rate=sampling_rate)
                if spike_idx is not None:
                    spikes.append((spike_idx, spike_val))
                if minima_idx is not None:
                    minimas.append((minima_idx, minima_val))

            # Collect all relevant information for this trial (exactly as in notebook)
            plot_data.append({
                'cell_ID': cell,
                'pre_post_status': pre_post_status,
                'trial_no': trial,
                'sampling_rate': sampling_rate,
                'cell_trace': trace,
                'field_trace': field_trace,
                'ttl_trace': ttl_trace,
                'ttl_rising_edges': ttl_rising_edges,
                'spikes': spikes,
                'minimas': minimas
            })
    
    # Save the plot data to a pickle file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(plot_data, f)
    
    print(f"CHR2 sensitisation plot data saved to {filename}")
    print(f"Total trials processed: {len(plot_data)}")
    
    # Debug information
    total_ttl_edges = sum(len(data['ttl_rising_edges']) for data in plot_data)
    total_spikes = sum(len(data['spikes']) for data in plot_data)
    total_minimas = sum(len(data['minimas']) for data in plot_data)
    
    print(f"Total TTL rising edges detected: {total_ttl_edges}")
    print(f"Total spikes detected: {total_spikes}")
    print(f"Total minimas detected: {total_minimas}")

def main():
    """Main function to parse arguments and generate sensitisation data pickle."""
    parser = argparse.ArgumentParser(description='Generate CHR2 sensitisation data pickle from training data')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input training data pickle file')
    parser.add_argument('--output', '-o', type=str, 
                        default='data/pickle_files/extract_features/pickle_files_from_analysis/sensitisation_plot_data.pkl',
                        help='Path to output pickle file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
    
    print(f"Loading training data from: {args.input}")
    try:
        raw_data_df = pd.read_pickle(args.input)
        print(f"Loaded data with {len(raw_data_df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Generate the sensitisation plot data
    save_plot_data_to_pickle(raw_data_df, args.output)

if __name__ == "__main__":
    main() 