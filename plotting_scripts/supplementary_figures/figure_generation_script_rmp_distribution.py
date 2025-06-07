__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzal.ks@gmail.com"

"""
Supplementary Figure: Resting Membrane Potential Distribution Analysis

This script generates a supplementary figure showing the distribution of resting membrane 
potentials across healthy cells (learners + non-learners only). The plot demonstrates 
that there's no inherent bias in RMP between different cell categories.

The figure includes:
- Panel A: Three violin plots: All Healthy Cells, Learners, Non-learners
- Panel B: RMP vs EPSP amplitude correlation (Pre-training)
- Panel C: RMP vs EPSP amplitude correlation (30mins post training)
- Panel D: RMP vs EPSP amplitude correlation with separate fits for learners/non-learners
- Panel E: RMP vs Firing frequency correlation (all healthy cells combined)
- Panel F: RMP vs Firing frequency correlation (Trainin data)
- Statistical comparisons and summary statistics for each analysis

Input files:
- cell_stats.h5: Cell statistics including RMP measurements
- all_cells_classified_dict.pickle: Cell classification data
- pd_all_cells_mean.pickle: EPSP amplitude data for correlation analysis
- all_cell_all_trial_firing_properties.pickle: Firing properties data
- sensitisation_plot_data.pickle: CHR2 sensitisation data for firing frequency calculation

Output: Figure_RMP_Distribution/rmp_distribution.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import scipy.stats as spst
from statannotations.Annotator import Annotator
import time
from pathlib import Path
import argparse
from matplotlib.gridspec import GridSpec
from shared_utils import baisic_plot_fuctnions_and_features as bpf
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set plot features using bpf
bpf.set_plot_properties()

class Args: pass
args_ = Args()

def plot_rmp_distribution_violin(cell_stats_with_category, fig, axs):
    """
    Create violin plots showing RMP distribution for all healthy cells, learners, and non-learners
    """
    # Filter to only include healthy cells (learners + non-learners)
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ].copy()
    
    # Create combined data for plotting
    plot_data_list = []
    
    # Add all healthy cells data
    all_healthy_rmp = healthy_cells['rmp'].values
    for rmp_val in all_healthy_rmp:
        plot_data_list.append({'rmp': rmp_val, 'category': 'All Healthy Cells'})
    
    # Add learners data
    learners_data = healthy_cells[healthy_cells['cell_type'] == 'learners']
    for _, row in learners_data.iterrows():
        plot_data_list.append({'rmp': row['rmp'], 'category': 'Learners'})
    
    # Add non-learners data
    non_learners_data = healthy_cells[healthy_cells['cell_type'] == 'non-learners']
    for _, row in non_learners_data.iterrows():
        plot_data_list.append({'rmp': row['rmp'], 'category': 'Non-learners'})
    
    # Convert to DataFrame
    plot_data = pd.DataFrame(plot_data_list)
    
    # Create violin plot with bpf colors
    palette = {
        'All Healthy Cells': bpf.CB_color_cycle[2],  # Green
        'Learners': bpf.CB_color_cycle[0],           # Blue
        'Non-learners': bpf.CB_color_cycle[1]        # Orange
    }
    
    g1 = sns.violinplot(
        data=plot_data,
        x="category", y="rmp",
        ax=axs,
        palette=palette,
        inner="quartile",
        order=['All Healthy Cells', 'Learners', 'Non-learners']
    )
    
    # Calculate and display summary statistics for all healthy cells
    rmp_mean = np.mean(all_healthy_rmp)
    rmp_std = np.std(all_healthy_rmp)
    rmp_median = np.median(all_healthy_rmp)
    n_cells = len(all_healthy_rmp)
    
    # Add text with statistics
    stats_text = f'All Healthy Cells:\nn = {n_cells}\nMean: {rmp_mean:.1f} ± {rmp_std:.1f} mV\nMedian: {rmp_median:.1f} mV'
    axs.text(0.02, 0.98, stats_text, transform=axs.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Statistical test between learners and non-learners only
    learners_rmp = learners_data['rmp'].values
    non_learners_rmp = non_learners_data['rmp'].values
    
    if len(learners_rmp) > 1 and len(non_learners_rmp) > 1:
        stat_test = spst.mannwhitneyu(learners_rmp, non_learners_rmp, nan_policy='omit')
        p_value = stat_test.pvalue
        
        # Add statistical annotation between learners and non-learners
        annotator = Annotator(axs, [("Learners", "Non-learners")],
                             data=plot_data, x="category", y="rmp")
        annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(p_value)])
        annotator.annotate()
    
    # Set labels and limits
    axs.set_ylabel("Resting Membrane\nPotential (mV)")
    axs.set_xlabel(None)
    axs.set_ylim(-80, -55)
    axs.tick_params(axis='x', rotation=45)
    
    # Remove top and right spines
    sns.despine(ax=axs, top=True, right=True)
    
    return axs

def plot_rmp_epsp_correlation(epsp_data, cell_stats_with_category, timepoint, fig, axs):
    """
    Plot correlation between RMP and EPSP amplitude for a specific timepoint (trained pattern only)
    """
    # Filter to only include healthy cells
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ].copy()
    
    # Get EPSP data for the specified timepoint and trained pattern only
    epsp_timepoint_data = epsp_data[
        (epsp_data['pre_post_status'] == timepoint) & 
        (epsp_data['frame_id'] == 'pattern_0')  # Trained pattern only
    ].copy()
    
    # Merge RMP and EPSP data
    correlation_data = []
    for _, cell_row in healthy_cells.iterrows():
        cell_id = cell_row['cell_ID']
        rmp = cell_row['rmp']
        
        # Get EPSP amplitude for this cell at the timepoint (trained pattern only)
        cell_epsp_data = epsp_timepoint_data[epsp_timepoint_data['cell_ID'] == cell_id]
        
        if not cell_epsp_data.empty:
            # Use all trials for this cell at this timepoint and pattern
            for _, trial_row in cell_epsp_data.iterrows():
                correlation_data.append({
                    'cell_ID': cell_id,
                    'rmp': rmp,
                    'epsp_amplitude': trial_row['max_trace'],
                    'cell_type': cell_row['cell_type']
                })
    
    if not correlation_data:
        axs.text(0.5, 0.5, f'No data available for {timepoint}', 
                transform=axs.transAxes, ha='center', va='center')
        return axs
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Create scatter plot without classification
    axs.scatter(correlation_df['rmp'], correlation_df['epsp_amplitude'], 
               color=bpf.CB_color_cycle[2], alpha=0.7, s=50)
    
    # Calculate correlation
    if len(correlation_df) > 2:
        correlation_coef, p_value = spst.pearsonr(correlation_df['rmp'], correlation_df['epsp_amplitude'])
        
        # Add correlation line
        z = np.polyfit(correlation_df['rmp'], correlation_df['epsp_amplitude'], 1)
        p = np.poly1d(z)
        axs.plot(correlation_df['rmp'], p(correlation_df['rmp']), 
                color=bpf.CB_color_cycle[3], linestyle='--', alpha=0.8)
        
        # Add correlation text
        corr_text = f'r = {correlation_coef:.3f}\np = {p_value:.3f}'
        if p_value < 0.05:
            corr_text += ' *'
        else:
            corr_text += ' (ns)'
            
        axs.text(0.02, 0.98, corr_text, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    axs.set_xlabel("Resting Membrane Potential (mV)")
    axs.set_ylabel("EPSP Amplitude (mV)")
    
    timepoint_label = 'Pre-training' if timepoint == 'pre' else f'30mins post training'
    axs.set_title(f'RMP vs EPSP Amplitude\n({timepoint_label})', fontsize=10)
    
    # Remove top and right spines
    sns.despine(ax=axs, top=True, right=True)
    
    return axs

def plot_rmp_epsp_learner_nonlearner(epsp_data, cell_stats_with_category, fig, axs):
    """
    Plot RMP vs EPSP correlation for learners and non-learners separately (post_3, trained pattern only)
    """
    # Filter to only include healthy cells
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ].copy()
    
    # Get EPSP data for post_3 and trained pattern only
    epsp_timepoint_data = epsp_data[
        (epsp_data['pre_post_status'] == 'post_3') & 
        (epsp_data['frame_id'] == 'pattern_0')  # Trained pattern only
    ].copy()
    
    # Separate data for learners and non-learners
    learner_data = []
    non_learner_data = []
    
    for _, cell_row in healthy_cells.iterrows():
        cell_id = cell_row['cell_ID']
        rmp = cell_row['rmp']
        cell_type = cell_row['cell_type']
        
        # Get EPSP amplitude for this cell at post_3 (trained pattern only)
        cell_epsp_data = epsp_timepoint_data[epsp_timepoint_data['cell_ID'] == cell_id]
        
        if not cell_epsp_data.empty:
            # Use all trials for this cell
            for _, trial_row in cell_epsp_data.iterrows():
                data_point = {
                    'cell_ID': cell_id,
                    'rmp': rmp,
                    'epsp_amplitude': trial_row['max_trace']
                }
                
                if cell_type == 'learners':
                    learner_data.append(data_point)
                else:  # non-learners
                    non_learner_data.append(data_point)
    
    if not learner_data and not non_learner_data:
        axs.text(0.5, 0.5, 'No data available for post_3', 
                transform=axs.transAxes, ha='center', va='center')
        return axs
    
    # Plot learners
    if learner_data:
        learner_df = pd.DataFrame(learner_data)
        axs.scatter(learner_df['rmp'], learner_df['epsp_amplitude'], 
                   color=bpf.CB_color_cycle[0], alpha=0.7, s=50, label='Learners')
        
        # Fit line for learners
        if len(learner_df) > 2:
            z_learners = np.polyfit(learner_df['rmp'], learner_df['epsp_amplitude'], 1)
            p_learners = np.poly1d(z_learners)
            axs.plot(learner_df['rmp'], p_learners(learner_df['rmp']), 
                    color=bpf.CB_color_cycle[0], linestyle='--', alpha=0.8)
            
            # Calculate correlation for learners
            corr_learners, p_learners_val = spst.pearsonr(learner_df['rmp'], learner_df['epsp_amplitude'])
            
            # Add correlation text for learners
            corr_text_learners = f'Learners: r = {corr_learners:.3f}'
            if p_learners_val < 0.05:
                corr_text_learners += ' *'
            else:
                corr_text_learners += ' (ns)'
    
    # Plot non-learners
    if non_learner_data:
        non_learner_df = pd.DataFrame(non_learner_data)
        axs.scatter(non_learner_df['rmp'], non_learner_df['epsp_amplitude'], 
                   color=bpf.CB_color_cycle[1], alpha=0.7, s=50, label='Non-learners')
        
        # Fit line for non-learners
        if len(non_learner_df) > 2:
            z_non_learners = np.polyfit(non_learner_df['rmp'], non_learner_df['epsp_amplitude'], 1)
            p_non_learners = np.poly1d(z_non_learners)
            axs.plot(non_learner_df['rmp'], p_non_learners(non_learner_df['rmp']), 
                    color=bpf.CB_color_cycle[1], linestyle='--', alpha=0.8)
            
            # Calculate correlation for non-learners
            corr_non_learners, p_non_learners_val = spst.pearsonr(non_learner_df['rmp'], non_learner_df['epsp_amplitude'])
            
            # Add correlation text for non-learners
            corr_text_non_learners = f'Non-learners: r = {corr_non_learners:.3f}'
            if p_non_learners_val < 0.05:
                corr_text_non_learners += ' *'
            else:
                corr_text_non_learners += ' (ns)'
    
    # Combine correlation text if both groups exist
    if learner_data and non_learner_data:
        combined_text = f'{corr_text_learners}\n{corr_text_non_learners}'
        axs.text(0.02, 0.98, combined_text, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif learner_data:
        axs.text(0.02, 0.98, corr_text_learners, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif non_learner_data:
        axs.text(0.02, 0.98, corr_text_non_learners, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    axs.set_xlabel("Resting Membrane Potential (mV)")
    axs.set_ylabel("EPSP Amplitude (mV)")
    axs.set_title('RMP vs EPSP Amplitude\n(30mins post training)', fontsize=10)
    
    # Add legend
    axs.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines
    sns.despine(ax=axs, top=True, right=True)
    
    return axs

def plot_rmp_firing_frequency_correlation(firing_properties, cell_stats_with_category, fig, axs):
    """
    Plot correlation between RMP and firing frequency for all healthy cells combined
    """
    # Filter to only include healthy cells
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ].copy()
    
    # Exclude the problematic cell from firing properties as well
    excluded_cell = "2022_12_12_cell_5"
    firing_properties_filtered = firing_properties[firing_properties['cell_ID'] != excluded_cell].copy()
    
    # Merge RMP and firing frequency data
    correlation_data = []
    for _, cell_row in healthy_cells.iterrows():
        cell_id = cell_row['cell_ID']
        rmp = cell_row['rmp']
        
        # Get firing frequency data for this cell
        cell_firing_data = firing_properties_filtered[firing_properties_filtered['cell_ID'] == cell_id]
        
        if not cell_firing_data.empty:
            # Use all firing frequency measurements for this cell
            for _, firing_row in cell_firing_data.iterrows():
                correlation_data.append({
                    'cell_ID': cell_id,
                    'rmp': rmp,
                    'firing_frequency': firing_row['spike_frequency'],
                    'injected_current': firing_row['injected_current']
                })
    
    if not correlation_data:
        axs.text(0.5, 0.5, 'No firing frequency data available', 
                transform=axs.transAxes, ha='center', va='center')
        return axs
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Create scatter plot with green color for combined healthy cells
    axs.scatter(correlation_df['rmp'], correlation_df['firing_frequency'], 
               color=bpf.CB_color_cycle[2], alpha=0.7, s=50, label='All Healthy Cells')
    
    # Calculate correlation
    if len(correlation_df) > 2:
        correlation_coef, p_value = spst.pearsonr(correlation_df['rmp'], correlation_df['firing_frequency'])
        
        # Add correlation line
        z = np.polyfit(correlation_df['rmp'], correlation_df['firing_frequency'], 1)
        p = np.poly1d(z)
        axs.plot(correlation_df['rmp'], p(correlation_df['rmp']), 
                color=bpf.CB_color_cycle[3], linestyle='--', alpha=0.8)
        
        # Add correlation text
        corr_text = f'r = {correlation_coef:.3f}\np = {p_value:.3f}'
        if p_value < 0.05:
            corr_text += ' *'
        else:
            corr_text += ' (ns)'
            
        axs.text(0.02, 0.98, corr_text, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add sample size info
    n_cells = len(correlation_df['cell_ID'].unique())
    n_measurements = len(correlation_df)
    info_text = f'n = {n_cells} cells\n({n_measurements} measurements)'
    axs.text(0.98, 0.02, info_text, transform=axs.transAxes, 
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    axs.set_xlabel("Resting Membrane Potential (mV)")
    axs.set_ylabel("Firing Frequency (Hz)")
    axs.set_title('RMP vs Firing Frequency\n(All Current Injections)', fontsize=10)
    
    # Remove top and right spines
    sns.despine(ax=axs, top=True, right=True)
    
    return axs

def plot_rmp_chr2_firing_frequency_correlation(sensitisation_data, cell_stats_with_category, fig, axs):
    """
    Plot correlation between RMP and firing frequency calculated from CHR2 sensitisation data
    """
    # Filter to only include healthy cells
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ].copy()
    
    # Exclude the problematic cell
    excluded_cell = "2022_12_12_cell_5"
    
    # Calculate firing frequency from CHR2 sensitisation data
    correlation_data = []
    
    # Group sensitisation data by cell_ID
    chr2_cell_data = {}
    for trial in sensitisation_data:
        cell_id = trial['cell_ID']
        if cell_id == excluded_cell:
            continue
            
        if cell_id not in chr2_cell_data:
            chr2_cell_data[cell_id] = []
        chr2_cell_data[cell_id].append(trial)
    
    for _, cell_row in healthy_cells.iterrows():
        cell_id = cell_row['cell_ID']
        rmp = cell_row['rmp']
        
        # Get CHR2 sensitisation data for this cell
        if cell_id in chr2_cell_data:
            for trial in chr2_cell_data[cell_id]:
                spikes = trial['spikes']
                sampling_rate = trial['sampling_rate']
                
                # Calculate firing frequency from spike count
                # Spikes are stored as [(time_index, amplitude), ...]
                if spikes and sampling_rate:
                    # Get recording duration from the first to last spike
                    spike_times = [spike[0] for spike in spikes]  # Extract time indices
                    if len(spike_times) >= 2:
                        duration_samples = max(spike_times) - min(spike_times)
                        duration_seconds = duration_samples / sampling_rate
                        
                        # Calculate firing frequency as spikes per second
                        if duration_seconds > 0:
                            firing_frequency = len(spikes) / duration_seconds
                        else:
                            # If all spikes occur at same time, estimate based on sampling
                            firing_frequency = len(spikes) * sampling_rate / 1000  # Assume 1ms window
                    else:
                        # Single spike or no spikes
                        firing_frequency = 0
                    
                    correlation_data.append({
                        'cell_ID': cell_id,
                        'rmp': rmp,
                        'firing_frequency': firing_frequency,
                        'trial_no': trial['trial_no']
                    })
    
    if not correlation_data:
        axs.text(0.5, 0.5, 'No Trainin frequency data available', 
                transform=axs.transAxes, ha='center', va='center')
        return axs
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Print debug information about CHR2 firing frequency data
    print(f"Trainin frequency data summary:")
    for cell_id in sorted(correlation_df['cell_ID'].unique()):
        cell_data = correlation_df[correlation_df['cell_ID'] == cell_id]
        freq_range = f"{cell_data['firing_frequency'].min():.1f} - {cell_data['firing_frequency'].max():.1f} Hz"
        print(f"  {cell_id}: {len(cell_data)} trials, {freq_range}")
    
    # Create scatter plot with different colors for each cell to identify outliers
    unique_cells = sorted(correlation_df['cell_ID'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cells)))  # Use tab10 colormap for distinct colors
    
    # Plot each cell with a different color
    for i, cell_id in enumerate(unique_cells):
        cell_data = correlation_df[correlation_df['cell_ID'] == cell_id]
        # Use short cell name for legend (last part after underscore)
        short_name = cell_id.split('_')[-1] if '_' in cell_id else cell_id
        axs.scatter(cell_data['rmp'], cell_data['firing_frequency'], 
                   color=colors[i], alpha=0.7, s=50, label=f'Cell {short_name}')
    
    # Calculate correlation
    if len(correlation_df) > 2:
        correlation_coef, p_value = spst.pearsonr(correlation_df['rmp'], correlation_df['firing_frequency'])
        
        # Add correlation line
        z = np.polyfit(correlation_df['rmp'], correlation_df['firing_frequency'], 1)
        p = np.poly1d(z)
        axs.plot(correlation_df['rmp'], p(correlation_df['rmp']), 
                color=bpf.CB_color_cycle[3], linestyle='--', alpha=0.8)
        
        # Add correlation text
        corr_text = f'r = {correlation_coef:.3f}\np = {p_value:.3f}'
        if p_value < 0.05:
            corr_text += ' *'
        else:
            corr_text += ' (ns)'
            
        axs.text(0.02, 0.98, corr_text, transform=axs.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add sample size info and outlier identification
    n_cells = len(correlation_df['cell_ID'].unique())
    n_measurements = len(correlation_df)
    
    # Identify potential outliers (values > 2 standard deviations from mean)
    mean_freq = correlation_df['firing_frequency'].mean()
    std_freq = correlation_df['firing_frequency'].std()
    outliers = correlation_df[abs(correlation_df['firing_frequency'] - mean_freq) > 2 * std_freq]
    
    info_text = f'n = {n_cells} cells\n({n_measurements} trials)'
    if len(outliers) > 0:
        outlier_cells = outliers['cell_ID'].unique()
        outlier_names = [cell.split('_')[-1] for cell in outlier_cells]
        info_text += f'\nOutliers: {", ".join(outlier_names)}'
    
    axs.text(0.98, 0.02, info_text, transform=axs.transAxes, 
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend with limited number of entries to avoid overcrowding
    if len(unique_cells) <= 8:  # Only show legend if not too many cells
        axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True)
    else:
        # For many cells, just add a note about color coding
        axs.text(0.02, 0.02, 'Each color = different cell', transform=axs.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    axs.set_xlabel("Resting Membrane Potential (mV)")
    axs.set_ylabel("Firing Frequency (Hz)")
    axs.set_title('RMP vs Firing Frequency\n(Training Data)', fontsize=10)
    
    # Remove top and right spines
    sns.despine(ax=axs, top=True, right=True)
    
    return axs

def create_rmp_distribution_figure(cell_stats_df, sc_data_dict, epsp_data, firing_properties, sensitisation_data, outdir):
    """
    Main function to create the RMP distribution figure with correlation analysis
    """
    # Process cell statistics data similar to Figure 3
    cell_stat_with_category = []
    
    # Get learner and non-learner cell IDs
    learner_cells = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learner_cells = sc_data_dict["an_cells"]["cell_ID"].unique()
    
    # Categorize cells into learners, non-learners, or other
    # Exclude problematic cell that gives offset values
    excluded_cell = "2022_12_12_cell_5"
    
    for cell in cell_stats_df.iterrows():
        cell_id = cell[0]
        
        # Skip the excluded cell
        if cell_id == excluded_cell:
            print(f"Excluding problematic cell: {cell_id}")
            continue
            
        rmp = cell[1]["cell_stats"]["rmp_median"]
        
        if cell_id in learner_cells:
            cell_type = "learners"
        elif cell_id in non_learner_cells:
            cell_type = "non-learners"
        else:
            cell_type = "other"
        
        cell_stat_with_category.append([cell_id, cell_type, rmp])
    
    # Convert to DataFrame
    c_cat_header = ["cell_ID", "cell_type", "rmp"]
    cell_stats_with_category = pd.concat([
        pd.DataFrame([i], columns=c_cat_header) for i in cell_stat_with_category
    ])
    
    # Filter to only healthy cells for summary
    healthy_cells = cell_stats_with_category[
        cell_stats_with_category['cell_type'].isin(['learners', 'non-learners'])
    ]
    
    print(f"RMP data summary (healthy cells only):")
    print(f"Total healthy cells: {len(healthy_cells)}")
    print(f"Learners: {len(healthy_cells[healthy_cells['cell_type'] == 'learners'])}")
    print(f"Non-learners: {len(healthy_cells[healthy_cells['cell_type'] == 'non-learners'])}")
    print(f"Excluded cells: {len(cell_stats_with_category[cell_stats_with_category['cell_type'] == 'other'])}")
    
    # Create figure with 6 subplots: RMP distribution, RMP vs EPSP (pre), RMP vs EPSP (post_3), learners vs non-learners (post_3), firing freq, CHR2 firing freq
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1, 1], height_ratios=[1, 1])
    gs.update(wspace=0.3, hspace=0.4)
    
    # Subplot A: RMP distribution violin plots
    axs_rmp = fig.add_subplot(gs[0, 0])
    plot_rmp_distribution_violin(cell_stats_with_category, fig, axs_rmp)
    axs_rmp.text(-0.1, 1.1, 'A', transform=axs_rmp.transAxes,
                 fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Subplot B: RMP vs EPSP correlation (pre)
    axs_pre = fig.add_subplot(gs[0, 1])
    plot_rmp_epsp_correlation(epsp_data, cell_stats_with_category, 'pre', fig, axs_pre)
    axs_pre.text(-0.1, 1.1, 'B', transform=axs_pre.transAxes,
                 fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Subplot C: RMP vs EPSP correlation (post_3)
    axs_post = fig.add_subplot(gs[0, 2])
    plot_rmp_epsp_correlation(epsp_data, cell_stats_with_category, 'post_3', fig, axs_post)
    axs_post.text(-0.1, 1.1, 'C', transform=axs_post.transAxes,
                  fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Subplot D: RMP vs EPSP correlation for learners vs non-learners (post_3)
    axs_learner_nonlearner = fig.add_subplot(gs[1, 0])
    plot_rmp_epsp_learner_nonlearner(epsp_data, cell_stats_with_category, fig, axs_learner_nonlearner)
    axs_learner_nonlearner.text(-0.1, 1.1, 'D', transform=axs_learner_nonlearner.transAxes,
                                fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Subplot E: RMP vs Firing frequency correlation
    axs_firing = fig.add_subplot(gs[1, 1])
    plot_rmp_firing_frequency_correlation(firing_properties, cell_stats_with_category, fig, axs_firing)
    axs_firing.text(-0.1, 1.1, 'E', transform=axs_firing.transAxes,
                    fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Subplot F: RMP vs CHR2 firing frequency correlation
    axs_chr2_firing = fig.add_subplot(gs[1, 2])
    plot_rmp_chr2_firing_frequency_correlation(sensitisation_data, cell_stats_with_category, fig, axs_chr2_firing)
    axs_chr2_firing.text(-0.1, 1.1, 'F', transform=axs_chr2_firing.transAxes,
                        fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Add main title
    fig.suptitle('Resting Membrane Potential Analysis in Healthy Cells', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    outpath = f"{outdir}/rmp_distribution.png"
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    print(f"Figure saved to: {outpath}")

def main():
    """Main function with argument parsing"""
    description = '''Generates RMP distribution supplementary figure with correlation analysis'''
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--cell-stats-path', '-c'
                        , required=False, default='./', type=str
                        , help='path to h5 file with cell stats')
    
    parser.add_argument('--classified-cells-path', '-s'
                        , required=False, default='./', type=str
                        , help='path to pickle file with classified cells')
    
    parser.add_argument('--epsp-data-path', '-f'
                        , required=False, default='./', type=str
                        , help='path to pickle file with EPSP data')
    
    parser.add_argument('--firing-properties-path', '-p'
                        , required=False, default='./', type=str
                        , help='path to pickle file with firing properties data')
    
    parser.add_argument('--sensitisation-data-path', '-x'
                        , required=False, default='./', type=str
                        , help='path to pickle file with CHR2 sensitisation data')
    
    parser.add_argument('--outdir-path', '-o'
                        , required=False, default='./', type=str
                        , help='where to save the generated figure image')
    
    args = parser.parse_args()
    cell_stats_path = Path(args.cell_stats_path)
    classified_cells_path = Path(args.classified_cells_path)
    epsp_data_path = Path(args.epsp_data_path)
    firing_properties_path = Path(args.firing_properties_path)
    sensitisation_data_path = Path(args.sensitisation_data_path)
    globoutdir = Path(args.outdir_path)
    
    # Create output directory
    globoutdir = globoutdir/'Figure_RMP_Distribution'
    globoutdir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading cell stats from: {cell_stats_path}")
    cell_stats_df = pd.read_hdf(cell_stats_path)
    
    print(f"Loading classified cells from: {classified_cells_path}")
    sc_data_dict = pd.read_pickle(classified_cells_path)
    
    print(f"Loading EPSP data from: {epsp_data_path}")
    epsp_data = pd.read_pickle(epsp_data_path)
    
    print(f"Loading firing properties from: {firing_properties_path}")
    firing_properties = pd.read_pickle(firing_properties_path)
    
    print(f"Loading sensitisation data from: {sensitisation_data_path}")
    sensitisation_data = pd.read_pickle(sensitisation_data_path)
    
    # Generate figure
    create_rmp_distribution_figure(cell_stats_df, sc_data_dict, epsp_data, firing_properties, sensitisation_data, globoutdir)

if __name__ == '__main__':
    # Timing the run
    ts = time.time()
    main()
    tf = time.time()
    print(f'Total time = {np.around(((tf-ts)/60), 1)} (mins)') 