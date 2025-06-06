#!/usr/bin/env python3
"""
CHR2 Sensitisation Supplementary Figure 6
Pattern Learning Paper - Supplementary Figures
Author: Anzal (anzal.ks@gmail.com)  
Repository: https://github.com/anzalks/

This script generates violin plots for CHR2 sensitisation analysis showing
spike amplitudes, TTL rising edges, and LFP minimas over time periods.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import argparse
import os
import sys

# Add the shared_utils directory to sys.path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
import baisic_plot_fuctnions_and_features as bpf

def load_sensitisation_data(filename):
    """
    Load CHR2 sensitisation data from pickle file.
    
    Parameters:
    filename (str): Path to pickle file
    
    Returns:
    list: Plot data loaded from pickle
    """
    try:
        with open(filename, 'rb') as f:
            plot_data = pickle.load(f)
        return plot_data
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading file: {e}")
        return None

def categorize_by_time(data_points):
    """
    Categorize data points into bins based on time.
    
    Parameters:
    data_points (list): List of (x, y) tuples
    
    Returns:
    dict: Categorized data by time points
    """
    time_points = ["0s", "0.5s", "1s"]
    categorized_data = {tp: [] for tp in time_points}
    
    if not data_points:
        return categorized_data
    
    x, y = zip(*data_points) if data_points else ([], [])
    unique_x = sorted(set(x))
    
    if not unique_x:
        return categorized_data
    
    # Categorize the points into the time bins
    for i, val in enumerate(x):
        if val < unique_x[len(unique_x) // 3]:
            categorized_data["0s"].append(y[i])
        elif val < unique_x[2 * len(unique_x) // 3]:
            categorized_data["0.5s"].append(y[i])
        else:
            categorized_data["1s"].append(y[i])
    
    return categorized_data

def create_dataframe(categorized_data):
    """
    Create a pandas DataFrame from categorized data.
    
    Parameters:
    categorized_data (dict): Dictionary with time points as keys
    
    Returns:
    DataFrame: Data formatted for plotting
    """
    return pd.DataFrame([
        {"Time": tp, "Value": v} 
        for tp, values in categorized_data.items() 
        for v in values
    ])

def plot_from_dataframe(ax, df, time_points, color='lightblue'):
    """
    Creates a violin plot with summary statistics.
    
    Parameters:
    ax: Matplotlib axis
    df: DataFrame with Time and Value columns
    time_points: List of time point labels
    color: Color for the violin plot
    """
    if df.empty:
        return
    
    sns.violinplot(data=df, x='Time', y='Value', inner='point', color=color, alpha=0.7, ax=ax)

    # Calculate and plot median, 25th, and 75th percentiles
    stats = df.groupby('Time')['Value'].describe(percentiles=[0.25, 0.5, 0.75])
    
    for i, tp in enumerate(time_points):
        if tp in stats.index:
            median = stats.loc[tp, '50%']
            q1 = stats.loc[tp, '25%']
            q3 = stats.loc[tp, '75%']
            # Plot median and quartiles as horizontal lines
            ax.plot([i - 0.2, i + 0.2], [median, median], color='black', lw=2)
            ax.plot([i - 0.2, i + 0.2], [q1, q1], color='grey', lw=1)
            ax.plot([i - 0.2, i + 0.2], [q3, q3], color='grey', lw=1)

def run_stat_tests(df):
    """
    Perform Mann-Whitney U tests between different time points.
    
    Parameters:
    df: DataFrame with Time and Value columns
    
    Returns:
    tuple: P-values for (0s vs 0.5s, 0.5s vs 1s, 0s vs 1s)
    """
    data_0s = df[df['Time'] == "0s"]['Value']
    data_05s = df[df['Time'] == "0.5s"]['Value']
    data_1s = df[df['Time'] == "1s"]['Value']

    def mannwhitney_safe(x, y):
        """Safe Mann-Whitney test handling small sample sizes."""
        if len(x) < 2 or len(y) < 2:
            return None
        return mannwhitneyu(x, y, alternative='two-sided').pvalue

    return (mannwhitney_safe(data_0s, data_05s),
            mannwhitney_safe(data_05s, data_1s),
            mannwhitney_safe(data_0s, data_1s))

def add_annotations(ax, p_values):
    """
    Adds annotations for p-values with asterisks.
    
    Parameters:
    ax: Matplotlib axis
    p_values: Tuple of p-values from statistical tests
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offsets = [y_max + 0.05 * y_range * (i + 1) for i in range(3)]
    comparisons = [(0, 1), (1, 2), (0, 2)]

    for (x1, x2), offset, p_val in zip(comparisons, offsets, p_values):
        if p_val is not None:
            # Draw a horizontal line between the groups
            ax.plot([x1, x2], [offset] * 2, color='black', linewidth=1.5)
            # Display asterisks and p-value next to the line
            ax.text((x1 + x2) / 2, offset + 0.01 * y_range, 
                   bpf.convert_pvalue_to_asterisks(p_val),
                   ha='center', va='bottom', fontsize=10, color='black')

def plot_chr2_sensitisation(filename, output_dir="outputs/supplementary_figures"):
    """
    Main plotting function for CHR2 sensitisation analysis.
    
    Parameters:
    filename (str): Path to sensitisation data pickle file
    output_dir (str): Directory to save the figure
    """
    # Load data
    plot_data = load_sensitisation_data(filename)
    if plot_data is None:
        return
    
    # Extract data from the pickle file
    ttl_rising_edge_points, spike_points, minima_points = [], [], []
    
    for data in plot_data:
        if all(k in data for k in ['ttl_trace', 'ttl_rising_edges', 'spikes', 'minimas']):
            ttl_trace = data['ttl_trace']
            ttl_rising_edges = data['ttl_rising_edges']
            spikes = data['spikes']
            minimas = data['minimas']
            ttl_rising_edge_points.extend([(edge, ttl_trace[edge]) for edge in ttl_rising_edges])
            spike_points.extend(spikes)
            minima_points.extend(minimas)

    # Define the time points
    time_points = ["0s", "0.5s", "1s"]

    # Prepare dataframes for plotting
    spike_df = create_dataframe(categorize_by_time(spike_points))
    ttl_df = create_dataframe(categorize_by_time(ttl_rising_edge_points))
    minima_df = create_dataframe(categorize_by_time(minima_points))

    # Create subplots and define colors (matching notebook exactly)
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    plot_colors = ['grey', 'grey', 'grey']  # Grey color as in notebook
    y_labels = ["Membrane Potential (mV)", "Trigger Voltage (V)", "LFP (mV)"]

    # Generate violin plots with annotations
    for ax, df, title, color, ylabel in zip(
        axs, [spike_df, ttl_df, minima_df], 
        ["Spike amplitudes", "TTL Rising Edges", "LFP minimas"], 
        plot_colors, y_labels
    ):
        plot_from_dataframe(ax, df, time_points, color)
        p_values = run_stat_tests(df)
        add_annotations(ax, p_values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(3))
        ax.set_xticklabels(time_points)
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Supplementary_figure_6_chr2_sensitisation.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Supplementary Figure 6 saved to: {output_path}")
    
    plt.show()

def main():
    """Main function to parse arguments and generate the supplementary figure."""
    parser = argparse.ArgumentParser(description='Generate CHR2 sensitisation supplementary figure')
    parser.add_argument('--sensitisation_data', '-s', type=str, required=True,
                        help='Path to sensitisation data pickle file')
    parser.add_argument('--output_dir', '-o', type=str, 
                        default='outputs/supplementary_figures',
                        help='Output directory for the figure')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sensitisation_data):
        print(f"Error: Sensitisation data file {args.sensitisation_data} does not exist.")
        sys.exit(1)
    
    # Generate the figure
    plot_chr2_sensitisation(args.sensitisation_data, args.output_dir)

if __name__ == "__main__":
    main() 