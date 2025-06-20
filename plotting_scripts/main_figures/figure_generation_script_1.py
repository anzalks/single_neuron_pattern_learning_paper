__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Figure 1: Experimental Setup and Basic Properties

This script generates Figure 1 of the pattern learning paper, which shows:
- Experimental setup with recording location and CA3 fluorescence
- Pattern stimulation grids (trained, overlapping, and non-overlapping patterns)
- Basic electrophysiological properties and traces
- Statistical analysis of baseline properties

Input files:
- baseline_traces_all_cells.pickle: Experimental data with baseline traces
- microscopy images: Setup and fluorescence images
- pd_all_cells_all_trials.pickle: All trial data for statistical analysis

Output: Figure_1/figure_1.png showing complete experimental setup and basic analysis
"""

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pickle
import PIL as pillow
from tqdm import tqdm
import numpy as np
import seaborn as sns
import scipy.stats as spst
import scipy
from statannotations.Annotator import Annotator
import time
from pathlib import Path
import argparse
from matplotlib.gridspec import GridSpec
from shared_utils import baisic_plot_fuctnions_and_features as bpf
from PIL import ImageDraw, ImageFont 
from scipy.stats import gaussian_kde

from scipy.stats import levene


# plot features are defines in bpf
bpf.set_plot_properties()

vlinec = "#C35817"

cell_to_plot = "2023_02_24_cell_2" 

time_to_plot = 0.15 # in s

class Args: pass
args_ = Args()



#def add_scale_bar_with_magnification(image_opened, scale_length_um, magnification, image_width_px, FoV_5x, bar_position=(50, 50), scale_bar_thickness=5, font_size=24):
#    """
#    Adds a white scale bar to a microscope image using Pillow and returns the image object.
#    
#    Parameters:
#    - image_opened: PIL Image object (already opened image).
#    - scale_length_um: float, length of the scale bar in micrometers.
#    - magnification: float, magnification of the microscope (e.g., 40x, 100x).
#    - image_width_px: int, width of the image in pixels.
#    - FoV_5x: float, field of view at 5x magnification in micrometers.
#    - bar_position: tuple (x, y), position of the scale bar's bottom-left corner.
#    - scale_bar_thickness: int, thickness of the scale bar in pixels.
#    - font_size: int, size of the font for the scale text.
#    
#    Returns:
#    - Image object with the scale bar.
#    """
#    # Draw on the image
#    draw = pillow.ImageDraw.Draw(image_opened)
#    
#    # Calculate the field of view (FoV) at the given magnification relative to 5x
#    FoV_magnification = FoV_5x * (5 / magnification)  # FoV in micrometers at the given magnification
#    
#    # Calculate pixels per micrometer at this magnification
#    pixels_per_um = image_width_px / FoV_magnification
#    
#    # Calculate scale bar length in pixels
#    scale_bar_length_px = int(scale_length_um * pixels_per_um)
#    
#    # Set position of the scale bar
#    bar_x = bar_position[0]  # X position in pixels
#    bar_y = image_opened.height - bar_position[1]  # Y position in pixels (from the bottom)
#    
#    # Draw the scale bar (a white rectangle)
#    draw.rectangle(
#        [bar_x, bar_y - scale_bar_thickness, bar_x + scale_bar_length_px, bar_y],
#        fill="white"
#    )
#    
#    # Add scale text
#    scale_text = f"{scale_length_um} μm"
#    
#    # Use a TrueType font if available
#    try:
#        # Specify the path to a TrueType font (use Arial as an example)
#        font_path = "/Library/Fonts/Arial.ttf"
#        font = pillow.ImageFont.truetype(font_path, font_size)
#    except IOError:
#        print("TrueType font not found. Using default bitmap font.")
#        font = pillow.ImageFont.load_default()
#
#    # Get the bounding box for the text
#    bbox = draw.textbbox((0, 0), scale_text, font=font)
#    text_width = bbox[2] - bbox[0]
#    text_height = bbox[3] - bbox[1]
#    
#    # Draw the text slightly above the scale bar
#    draw.text(
#        (bar_x, bar_y - scale_bar_thickness - text_height - 5),  # Position slightly above the scale bar
#        scale_text,
#        fill="white",
#        font=font
#    )
#    
#    return image_opened




def add_scale_bar_with_magnification(image_opened, scale_length_um, magnification, image_width_px, 
                                     bar_position=(50, 50), scale_bar_thickness=5, font_size=24):
    """
    Adds a white scale bar to a microscope image using Pillow and returns the image object.
    
    Parameters:
    - image_opened: PIL Image object (already opened image).
    - scale_length_um: float, length of the scale bar in micrometers.
    - magnification: float, magnification of the microscope (e.g., 5x, 40x).
    - image_width_px: int, width of the image in pixels.
    - bar_position: tuple (x, y), position of the scale bar's bottom-left corner.
    - scale_bar_thickness: int, thickness of the scale bar in pixels.
    - font_size: int, size of the font for the scale text.
    
    Returns:
    - Image object with the scale bar.
    """
    # Base um_per_pixel at 40x magnification
    um_per_pixel_40x = 0.1961  # micrometers per pixel at 40x

    # Calculate um_per_pixel for the given magnification
    um_per_pixel = (40 / magnification) * um_per_pixel_40x

    # Calculate scale bar length in pixels
    scale_bar_length_px = int(scale_length_um / um_per_pixel)
    
    # Draw on the image
    draw = pillow.ImageDraw.Draw(image_opened)
    
    # Set position of the scale bar
    bar_x = bar_position[0]  # X position in pixels
    bar_y = image_opened.height - bar_position[1]  # Y position in pixels (from the bottom)
    
    # Draw the scale bar (a white rectangle)
    draw.rectangle(
        [bar_x, bar_y - scale_bar_thickness, bar_x + scale_bar_length_px, bar_y],
        fill="white"
    )
    
    # Add scale text
    scale_text = f"{scale_length_um} μm"
    
    # Use a TrueType font if available
    try:
        # Specify the path to a TrueType font (use Arial as an example)
        font_path = "/Library/Fonts/Arial.ttf"
        font = pillow.ImageFont.truetype(font_path, font_size)
    except IOError:
        print("TrueType font not found. Using default bitmap font.")
        font = pillow.ImageFont.load_default()
    
    # Add text near the scale bar
    text_x = bar_x
    text_y = bar_y - scale_bar_thickness - font_size - 10  # Adding some spacing between bar and text
    draw.text((text_x, text_y), scale_text, font=font, fill="white")
    
    return image_opened














































#def add_scale_bar_with_magnification(image_opened, magnification, image_width_px, bar_position=(50, 50), scale_bar_thickness=5, font_size=24):
#    """
#    Adds a white scale bar to a microscope image using Pillow and returns the image object.
#
#    Parameters:
#    - image_opened: PIL Image object (already opened image).
#    - magnification: float, magnification of the microscope (e.g., 4x, 40x).
#    - image_width_px: int, width of the image in pixels.
#    - bar_position: tuple (x, y), position of the scale bar's bottom-left corner.
#    - scale_bar_thickness: int, thickness of the scale bar in pixels.
#    - font_size: int, size of the font for the scale text.
#
#    Returns:
#    - Image object with the scale bar.
#    """
#    # Previously calculated µm per pixel for 4x and 40x
#    if magnification == 40:
#        um_per_pixel = 0.1961  # µm per pixel at 40x
#    elif magnification == 4:
#        um_per_pixel = 1.961  # µm per pixel at 4x
#    else:
#        raise ValueError("Unsupported magnification. Please use 4x or 40x.")
#
#    # Draw on the image
#    draw = ImageDraw.Draw(image_opened)
#    
#    # Set scale length (100 µm for example)
#    scale_length_um = 100  # You can adjust the scale length based on the image
#    
#    # Calculate scale bar length in pixels
#    scale_bar_length_px = int(scale_length_um / um_per_pixel)
#    
#    # Set position of the scale bar
#    bar_x = bar_position[0]  # X position in pixels
#    bar_y = image_opened.height - bar_position[1]  # Y position in pixels (from the bottom)
#    
#    # Draw the scale bar (a white rectangle)
#    draw.rectangle(
#        [bar_x, bar_y - scale_bar_thickness, bar_x + scale_bar_length_px, bar_y],
#        fill="white"
#    )
#    
#    # Add scale text
#    scale_text = f"{scale_length_um} μm"
#    
#    # Use a TrueType font if available
#    try:
#        # Specify the path to a TrueType font (use Arial as an example)
#        font_path = "/Library/Fonts/Arial.ttf"  # Adjust the path as per your environment
#        font = ImageFont.truetype(font_path, font_size)
#    except IOError:
#        print("TrueType font not found. Using default bitmap font.")
#        font = ImageFont.load_default()
#
#    # Get the bounding box for the text
#    bbox = draw.textbbox((0, 0), scale_text, font=font)
#    text_width = bbox[2] - bbox[0]
#    text_height = bbox[3] - bbox[1]
#    
#    # Draw the text slightly above the scale bar
#    draw.text(
#        (bar_x, bar_y - scale_bar_thickness - text_height - 5),  # Position slightly above the scale bar
#        scale_text,
#        fill="white",
#        font=font
#    )
#    
#    return image_opened




def add_labels_to_image(image, text_list, coordinates_list, font_size=20,
                        font_color=(255, 255, 255)):
    """
    Adds text labels to an image at specified coordinates.
    
    :param image: Image object returned by PIL.Image.open.
    :param text_list: List of strings to add to the image as labels.
    :param coordinates_list: List of tuples with (x, y) coordinates for each text label.
    :param font_size: Font size of the labels. Default is 20.
    :param font_color: Font color of the labels as an RGB tuple. Default is red (255, 0, 0).
    :return: Image object with text labels added.
    """
    # Make the image editable
    draw = ImageDraw.Draw(image)
    
    # Define a font (you can specify a font file if available, else it will use default)
    try:
        # Specify the path to a TrueType font (use Arial as an example)
        font_path = "/Library/Fonts/Arial.ttf"
        font = pillow.ImageFont.truetype(font_path, font_size)
    except IOError:
        print("TrueType font not found. Using default bitmap font.")
        font = pillow.ImageFont.load_default()
    # Add text labels at specified coordinates
    for text, (x, y) in zip(text_list, coordinates_list):
        draw.text((x, y), text, font=font, fill=font_color)
    
    return image


def int_to_roman(num):
    # Helper function to convert integer to Roman numeral
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "iX", "V", "iV",
        "i"
        ]
    roman_num = ''
    i = 0
    while  num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def label_axis(axis_list, letter_label, xpos=0.1, ypos=1, fontsize=16, fontweight='bold'):
    for axs_no, axs in enumerate(axis_list):
        roman_no = int_to_roman(axs_no + 1)  # Convert number to Roman numeral
        axs.text(xpos, ypos, f'{letter_label}{roman_no}', 
                 transform=axs.transAxes, fontsize=fontsize, 
                 fontweight=fontweight, ha='center', va='center')

def move_axis(axs_list,xoffset,yoffset,pltscale):
    for axs in axs_list:
        pos = axs.get_position()  # Get the original position
        new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
                   pos.height*pltscale]
        # Shrink the plot
        axs.set_position(new_pos)

def plot_image(image,axs_img,xoffset,yoffset,pltscale):
    axs_img.imshow(image, cmap='gray')
    pos = axs_img.get_position()  # Get the original position
    new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
               pos.height*pltscale]
    # Shrink the plot
    axs_img.set_position(new_pos)
    axs_img.axis('off')

def plot_patterns(axs_pat1,axs_pat2,axs_pat3,xoffset,yoffset):
    pattern_list = ["trained pattern","Overlapping pattern",
                    "Non-overlapping pattern"]
    for pr_no, pattern in enumerate(pattern_list):
        if pr_no==0:
            axs_pat = axs_pat1  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(0,1.45)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,1.45)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,1.45)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        pat_pos = axs_pat.get_position()
        new_pat_pos = [pat_pos.x0+xoffset, pat_pos.y0+yoffset, pat_pos.width,
                        pat_pos.height]
        axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        axs_pat.set_title(pattern,fontsize=10)

def plot_trace_raw_all_pats(cell_data,field_to_plot,ylim,xlim,
                            ylabel,axs1,axs2,axs3):
    sampling_rate = int(cell_data["sampling_rate(Hz)"].unique())
    axs = [axs1,axs2,axs3]
    pattern_grps = cell_data.groupby(by="frame_id")
    for pat,pat_data in pattern_grps:
        axs_no = int(pat.split('_')[-1])
        trial_grp = pat_data.groupby(by="trial_no")
        mean_trace = []
        for tr, trial_data in trial_grp:
            trace = trial_data[field_to_plot].to_numpy()[:int(time_to_plot*sampling_rate)]
            trace = bpf.substract_baseline(trace,bl_period_in_ms=2)
            mean_trace.append(trace)
            time = np.linspace(0,time_to_plot,len(trace))*1000
            axs[axs_no].plot(time, trace, color='k',alpha=0.2, label="trials")
        mean_trace = np.array(mean_trace)
        mean_trace = np.mean(mean_trace,axis=0)
        time = np.linspace(0,time_to_plot,len(mean_trace))*1000
        axs[axs_no].plot(time,mean_trace,color='k',label="mean")
        axs[axs_no].axvline(0,color=vlinec,linestyle=':',label="optical\nstim")
        axs[axs_no].spines[['right', 'top']].set_visible(False)
        axs[axs_no].set_ylim(ylim)
        axs[axs_no].set_xlim(-10,xlim)
        axs[axs_no].set_xlabel("time (ms)")
        axs[axs_no].set_ylabel(ylabel)

def inset_plot_traces(cell_data,field_to_plot,
                      axs, pat_num,
                      ylim=(-0.7,0.3),xlim=15,
                      xoffset=0.15,yoffset=0.015,
                      pltscale=0.35):
    sampling_rate = int(cell_data["sampling_rate(Hz)"].unique())
    pattern_grps = cell_data.groupby(by="frame_id")
    for pat,pat_data in pattern_grps:
        pat_no = int(pat.split('_')[-1])
        if pat_no!=pat_num:
            continue
        elif pat_no==pat_num:
            trial_grp = pat_data.groupby(by="trial_no")
            mean_trace = []
            for tr, trial_data in trial_grp:
                trace = trial_data[field_to_plot].to_numpy()[:int(time_to_plot*sampling_rate)]
                trace = bpf.substract_baseline(trace,bl_period_in_ms=2)
                mean_trace.append(trace)
                time = np.linspace(0,time_to_plot,len(trace))*1000
                axs.plot(time, trace, color='k',alpha=0.2)
            mean_trace = np.array(mean_trace)
            mean_trace = np.mean(mean_trace,axis=0)
            time = np.linspace(0,time_to_plot,len(mean_trace))*1000
            axs.plot(time,mean_trace,color='k')
            axs.axvline(0,color=vlinec,linestyle=':')
            axs.spines[['right', 'top']].set_visible(False)
            axs.set_ylim(ylim)
            axs.set_xlim(0,xlim)
            axs.set_xlabel(None)
            axs.set_ylabel(None)
            #axs.set_xticks([])
            #axs.set_yticks([])
            #axs.set_yticklabels([])
            #axs.set_xticklabels([])
            
            pos = axs.get_position()  # Get the original position
            new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale, 
                    pos.height*pltscale ]
            axs.set_position(new_pos)
        else:
            continue



#def plot_trace_stats(feature_extracted_df, fig, axs_cell, axs_field):
#    """
#    Plot normalized KDE distributions of `max_trace` and `min_field` 
#    after normalizing their mean to 1.
#    
#    Parameters:
#    - feature_extracted_df (pd.DataFrame): Input DataFrame containing features.
#    - fig (plt.Figure): Figure object.
#    - axs_cell (plt.Axes): Axis for `max_trace` plot.
#    - axs_field (plt.Axes): Axis for `min_field` plot.
#    """
#    
#    # Check if required columns exist in the DataFrame
#    required_cols = ['pre_post_status', 'trial_no', 'cell_ID', 'frame_id', 'max_trace', 'min_field']
#    if not all(col in feature_extracted_df.columns for col in required_cols):
#        raise ValueError("Input DataFrame is missing required columns.")
#    
#    # Step 1: Filter data for 'pre' status
#    filtered_df = feature_extracted_df[feature_extracted_df['pre_post_status'] == 'pre']
#    
#    # Step 2: Calculate the mean for trials 0, 1, and 2, grouped by cell_ID and frame_id
#    means_df = filtered_df[
#        filtered_df['trial_no'].isin([0, 1, 2])
#    ].groupby(['cell_ID', 'frame_id'])[['max_trace', 'min_field']].mean().reset_index()
#    
#    # Step 3: Normalize `max_trace` and `min_field` such that their mean is 1
#    means_df['max_trace'] /= means_df['max_trace'].mean() if means_df['max_trace'].mean() != 0 else 1
#    means_df['min_field'] /= means_df['min_field'].mean() if means_df['min_field'].mean() != 0 else 1
#
#    # Helper function to compute and plot normalized KDE
#    def plot_normalized_kde(ax, data, color, label):
#        if len(data) > 1:
#            kde = gaussian_kde(data)
#            x_vals = np.linspace(min(data), max(data), 1000)
#            kde_vals = kde(x_vals)
#            
#            # Normalize KDE peak to 1
#            max_val = np.max(kde_vals)
#            if max_val != 0:
#                kde_vals /= max_val
#            
#            # Plot the normalized KDE
#            ax.fill_between(x_vals, kde_vals, color=color, alpha=0.3)
#            ax.plot(x_vals, kde_vals, color=color, linewidth=1)
#        else:
#            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
#        
#        ax.set_ylim(0, 1)
#        ax.set_xlabel(f'Normalized Mean {label}')
#        ax.set_ylabel('Normalized Density')
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        ax.set_title(f'Normalized Distribution of {label}')
#    
#    # Step 4: Plot KDE for `max_trace` on axs_cell
#    plot_normalized_kde(axs_cell, means_df['max_trace'], color='blue', label='max_trace')
#    
#    # Step 5: Plot KDE for `min_field` on axs_field
#    plot_normalized_kde(axs_field, means_df['min_field'], color='orange', label='min_field')
#    
#    # Improve layout
#    fig.tight_layout()

# Updated function to include standard deviation display on the plot
def plot_bar_distribution(ax, data, color, label, show_ylabel=True, remove_ytick_labels=False):
    if len(data) > 1:
        # Calculate the mean and standard deviation
        mean_val = data.mean()
        std_val = data.std()
        
        # Plot the histogram as a bar plot (without normalizing the counts)
        n, bins, patches = ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor='black')
        
        # Overlay the mean line and shaded region for standard deviation
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(
            y=[0, n.max()],
            x1=mean_val - std_val,
            x2=mean_val + std_val,
            color='red',
            alpha=0.3
        )
        
        # Add text to show the standard deviation
        ax.text(
            mean_val + std_val, n.max() * 0.9, 
            f"σ = {std_val:.2f}", 
            color='red', 
            fontsize=10,
            va='center'
        )
        
        ax.set_xlabel(f'Normalized\n{label}')
        if show_ylabel:
            ax.set_ylabel('Count')
        if remove_ytick_labels:
            ax.set_yticklabels([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return n.max(), bins
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
        return 0, None


def plot_violin_distribution(ax, data, color, label, show_ylabel=True, remove_ytick_labels=False):
    """
    Plot vertical violin distribution to match the exact layout shown in the user's figure.
    The violin extends horizontally from a central vertical axis, with data values on the y-axis.
    """
    if len(data) > 1:
        # Calculate the mean and standard deviation
        mean_val = data.mean()
        std_val = data.std()
        
        # Create vertical violin plot - default orientation (vert=True)
        violin_parts = ax.violinplot([data], positions=[0], vert=True, 
                                   showmeans=True, showmedians=True, widths=0.8)
        
        # Style the violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style the statistical lines
        if 'cmeans' in violin_parts:
            violin_parts['cmeans'].set_color('red')
            violin_parts['cmeans'].set_linewidth(2)
        if 'cmedians' in violin_parts:
            violin_parts['cmedians'].set_color('blue')
            violin_parts['cmedians'].set_linewidth(1.5)
        if 'cmaxes' in violin_parts:
            violin_parts['cmaxes'].set_color('black')
            violin_parts['cmaxes'].set_linewidth(1)
        if 'cmins' in violin_parts:
            violin_parts['cmins'].set_color('black')
            violin_parts['cmins'].set_linewidth(1)
        if 'cbars' in violin_parts:
            violin_parts['cbars'].set_color('black')
            violin_parts['cbars'].set_linewidth(1)
        
        # Add text to show the standard deviation
        ax.text(
            0.3, max(data) * 0.9, 
            f"σ = {std_val:.2f}", 
            color='red', 
            fontsize=10,
            va='center',
            ha='left'
        )
        
        # Set labels and formatting
        ax.set_xlabel(f'Normalized\n{label}')
        ax.set_xticks([0])
        ax.set_xticklabels([''])
        
        if show_ylabel:
            ax.set_ylabel('Distribution')
        else:
            ax.set_ylabel('')
            
        if remove_ytick_labels:
            ax.set_yticklabels([])
        
        # Remove spines for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Set x-axis limits to center the violin
        ax.set_xlim(-0.5, 0.5)
        
        return max(data), min(data)
    else:
        ax.text(0, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
        return 0, None

def plot_trace_stats(feature_extracted_df, fig, ax_single): 
    """
    Plot a single vertical violin plot comparing EPSP and LFP distributions on one axis (Panel E).
    
    Parameters:
    - feature_extracted_df (pd.DataFrame): Input DataFrame containing features.
    - fig (plt.Figure): Figure object.
    - ax_single (plt.Axes): Single axis for the combined violin plot.
    """
    required_cols = ['pre_post_status', 'trial_no', 'cell_ID', 'frame_id', 
                     'max_trace', 'min_field', 'max_trace_t', 'min_field_t']
    if not all(col in feature_extracted_df.columns for col in required_cols):
        raise ValueError("Input DataFrame is missing required columns.")
    
    filtered_df = feature_extracted_df[feature_extracted_df['pre_post_status'] == 'pre']
    mean_combined_df = filtered_df[
        filtered_df['trial_no'].isin([0, 1, 2])
    ].groupby(['cell_ID', 'frame_id'])[['max_trace', 'min_field', 'max_trace_t', 'min_field_t']].mean().reset_index()
    
    merged_df = filtered_df.merge(mean_combined_df, on=['cell_ID', 'frame_id'], suffixes=('', '_mean'))
    merged_df['max_trace'] = merged_df['max_trace'] / merged_df['max_trace_mean']
    merged_df['min_field'] = merged_df['min_field'] / merged_df['min_field_mean']
    merged_df['max_trace_t'] = merged_df['max_trace_t'] / merged_df['max_trace_t_mean']
    merged_df['min_field_t'] = merged_df['min_field_t'] / merged_df['min_field_t_mean']
    
    # Create a DataFrame for plotting both EPSP and LFP on the same plot
    data_to_plot = pd.DataFrame({
        'Normalized': np.concatenate([merged_df['max_trace'].dropna(), merged_df['min_field'].dropna()]),
        'Category': ['EPSP'] * len(merged_df['max_trace'].dropna()) + ['LFP'] * len(merged_df['min_field'].dropna())
    })
    
    # Create clean violin plot without error bars, quartiles, or extrema - just like the reference figure
    violin_parts = ax_single.violinplot([merged_df['max_trace'].dropna(), merged_df['min_field'].dropna()], 
                                       positions=[0, 1], vert=True, showmeans=False, showmedians=False, 
                                       showextrema=False, widths=0.6)
    
    # Style the violin plot with gray/black tones to match the figure
    colors = ['gray', 'black']  # Gray for EPSP, black for LFP
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Add mean and SD as dotted lines that respect violin boundaries
    mean_epsp = merged_df['max_trace'].dropna().mean()
    std_epsp = merged_df['max_trace'].dropna().std()
    mean_lfp = merged_df['min_field'].dropna().mean()
    std_lfp = merged_df['min_field'].dropna().std()
    
    # Get the violin shapes to calculate proper line lengths
    epsp_data = merged_df['max_trace'].dropna()
    lfp_data = merged_df['min_field'].dropna()
    
    # Calculate violin widths at mean and SD positions using kernel density
    from scipy.stats import gaussian_kde
    
    # For EPSP violin (position 0)
    kde_epsp = gaussian_kde(epsp_data)
    
    # Calculate relative widths at mean and SD positions
    width_at_mean_epsp = kde_epsp(mean_epsp)[0]
    width_at_sd_high_epsp = kde_epsp(mean_epsp + std_epsp)[0] if (mean_epsp + std_epsp) <= epsp_data.max() else 0
    width_at_sd_low_epsp = kde_epsp(mean_epsp - std_epsp)[0] if (mean_epsp - std_epsp) >= epsp_data.min() else 0
    
    # Normalize widths (violin default width is 0.6, so max extension is 0.3 on each side)
    max_width_epsp = kde_epsp(epsp_data).max()
    norm_width_mean_epsp = (width_at_mean_epsp / max_width_epsp) * 0.25
    norm_width_sd_high_epsp = (width_at_sd_high_epsp / max_width_epsp) * 0.25 if width_at_sd_high_epsp > 0 else 0
    norm_width_sd_low_epsp = (width_at_sd_low_epsp / max_width_epsp) * 0.25 if width_at_sd_low_epsp > 0 else 0
    
    # For LFP violin (position 1)
    kde_lfp = gaussian_kde(lfp_data)
    
    width_at_mean_lfp = kde_lfp(mean_lfp)[0]
    width_at_sd_high_lfp = kde_lfp(mean_lfp + std_lfp)[0] if (mean_lfp + std_lfp) <= lfp_data.max() else 0
    width_at_sd_low_lfp = kde_lfp(mean_lfp - std_lfp)[0] if (mean_lfp - std_lfp) >= lfp_data.min() else 0
    
    max_width_lfp = kde_lfp(lfp_data).max()
    norm_width_mean_lfp = (width_at_mean_lfp / max_width_lfp) * 0.25
    norm_width_sd_high_lfp = (width_at_sd_high_lfp / max_width_lfp) * 0.25 if width_at_sd_high_lfp > 0 else 0
    norm_width_sd_low_lfp = (width_at_sd_low_lfp / max_width_lfp) * 0.25 if width_at_sd_low_lfp > 0 else 0
    
    # Add dotted lines that respect violin boundaries
    # Mean lines
    ax_single.hlines(mean_epsp, 0 - norm_width_mean_epsp, 0 + norm_width_mean_epsp, colors='black', linestyles='dotted', linewidth=1.5)
    ax_single.hlines(mean_lfp, 1 - norm_width_mean_lfp, 1 + norm_width_mean_lfp, colors='black', linestyles='dotted', linewidth=1.5)
    
    # SD lines (only draw if within data range)
    if norm_width_sd_high_epsp > 0:
        ax_single.hlines(mean_epsp + std_epsp, 0 - norm_width_sd_high_epsp, 0 + norm_width_sd_high_epsp, colors='black', linestyles='dotted', linewidth=1)
    if norm_width_sd_low_epsp > 0:
        ax_single.hlines(mean_epsp - std_epsp, 0 - norm_width_sd_low_epsp, 0 + norm_width_sd_low_epsp, colors='black', linestyles='dotted', linewidth=1)
    if norm_width_sd_high_lfp > 0:
        ax_single.hlines(mean_lfp + std_lfp, 1 - norm_width_sd_high_lfp, 1 + norm_width_sd_high_lfp, colors='black', linestyles='dotted', linewidth=1)
    if norm_width_sd_low_lfp > 0:
        ax_single.hlines(mean_lfp - std_lfp, 1 - norm_width_sd_low_lfp, 1 + norm_width_sd_low_lfp, colors='black', linestyles='dotted', linewidth=1)
    
    # Set labels and formatting to match the figure
    ax_single.set_ylabel('Normalized\nmean')
    ax_single.set_xticks([0, 1])
    ax_single.set_xticklabels(['EPSP', 'LFP'])
    ax_single.set_xlabel('')  # Remove x-axis label to match figure
    
    # Remove spines for clean look
    ax_single.spines['top'].set_visible(False)
    ax_single.spines['right'].set_visible(False)
    
    # Set y-axis limits to match the figure (0 to 3)
    ax_single.set_ylim(0, 3)
    
    # Add Levene's test statistical annotation with line for panel E
    from scipy.stats import levene
    stat, p_value = levene(merged_df['max_trace'].dropna(), merged_df['min_field'].dropna())
    
    # Convert p-value to asterisks and add annotation
    annotation = bpf.convert_pvalue_to_asterisks(p_value)
    
    # Add statistical annotation line between the violins
    y_max = 2.7
    ax_single.plot([0, 0, 1, 1], [y_max-0.1, y_max, y_max, y_max-0.1], 'k-', linewidth=1)
    
    # Add significance stars in black
    ax_single.text(0.5, y_max + 0.1, annotation, 
                   fontsize=14, ha='center', va='bottom', 
                   color='black', fontweight='bold')

def plot_trace_stats_with_pvalue(feature_extracted_df, fig, ax_left, ax_right):
    """
    Plot bar plots (histograms) of peak timing data (`max_trace_t` and `min_field_t`) on separate axes (Fi and Fii).
    
    Parameters:
    - feature_extracted_df (pd.DataFrame): Input DataFrame containing features.
    - fig (plt.Figure): Figure object.
    - ax_left (plt.Axes): Axis for the EPSP timing histogram (Fi).
    - ax_right (plt.Axes): Axis for the LFP timing histogram (Fii).
    """
    
    # Check if required columns exist in the DataFrame
    required_cols = ['pre_post_status', 'trial_no', 'cell_ID', 'frame_id', 'max_trace_t', 'min_field_t']
    if not all(col in feature_extracted_df.columns for col in required_cols):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Step 1: Filter data for 'pre' status
    filtered_df = feature_extracted_df[feature_extracted_df['pre_post_status'] == 'pre']
    
    # Step 2: Calculate the mean across trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`
    mean_combined_df = filtered_df[
        filtered_df['trial_no'].isin([0, 1, 2])
    ].groupby(['cell_ID', 'frame_id'])[['max_trace_t', 'min_field_t']].mean().reset_index()
    
    # Step 3: Merge the combined mean back to the original filtered data
    merged_df = filtered_df.merge(mean_combined_df, on=['cell_ID', 'frame_id'], suffixes=('', '_mean'))
    
    # Step 4: Normalize timing data using the combined mean
    merged_df['max_trace_t'] = merged_df['max_trace_t'] / merged_df['max_trace_t_mean']
    merged_df['min_field_t'] = merged_df['min_field_t'] / merged_df['min_field_t_mean']
    
    # Step 5: Create bar plots with gray bars and red SD overlays to match the figure
    # Left plot (Fi) - EPSP timing
    data_left = merged_df['max_trace_t'].dropna()
    mean_left = data_left.mean()
    std_left = data_left.std()
    
    n_left, bins_left, patches_left = ax_left.hist(data_left, bins=25, color='darkgray', alpha=0.8, edgecolor='black')
    # Add SD indication with red shaded area
    ax_left.fill_betweenx(y=[0, n_left.max()], x1=mean_left - std_left, x2=mean_left + std_left, 
                         color='red', alpha=0.3)
    # Add sigma text
    ax_left.text(mean_left + std_left, n_left.max() * 0.9, f"σ = {std_left:.2f}", 
                color='red', fontsize=10, va='center')
    
    ax_left.set_xlabel('time (ms)\n(EPSP)')
    ax_left.set_ylabel('Count')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    
    # Right plot (Fii) - LFP timing  
    data_right = merged_df['min_field_t'].dropna()
    mean_right = data_right.mean()
    std_right = data_right.std()
    
    n_right, bins_right, patches_right = ax_right.hist(data_right, bins=25, color='darkgray', alpha=0.8, edgecolor='black')
    # Add SD indication with red shaded area
    ax_right.fill_betweenx(y=[0, n_right.max()], x1=mean_right - std_right, x2=mean_right + std_right, 
                          color='red', alpha=0.3)
    # Add sigma text
    ax_right.text(mean_right + std_right, n_right.max() * 0.9, f"σ = {std_right:.2f}", 
                 color='red', fontsize=10, va='center')
    
    ax_right.set_xlabel('time (ms)\n(LFP)')
    ax_right.set_ylabel('')
    ax_right.set_yticklabels([])
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    # Remove p-value annotations to match the clean figure
    # (No statistical annotations in Fi and Fii panels)
    
    # Set consistent y-axis limits for both bar plots
    max_count = max(n_left.max(), n_right.max())
    ax_left.set_ylim(0, max_count * 1.05)
    ax_right.set_ylim(0, max_count * 1.05)

def plot_figure_1(pickle_file_path,
                  all_trials_path,
                  image_file_path,
                  projection_image,
                  outdir,cell_to_plot=cell_to_plot):
    cell_data = pd.read_pickle(pickle_file_path)
    alltrial_Df=pd.read_pickle(all_trials_path)
    print(f"dataframe:{alltrial_Df.columns}...........")
    deselect_lsit = ["no_frame","inR"]
    cell_data = cell_data[(cell_data["cell_ID"]==cell_to_plot)&(~cell_data["frame_id"].isin(deselect_lsit))]
    cell_data.reset_index()
    sampling_rate = int(cell_data["sampling_rate(Hz)"].unique())
    image= pillow.Image.open(image_file_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image= add_scale_bar_with_magnification(image_opened=image,
                                            magnification=4,
                                            scale_length_um=500, 
                                            image_width_px=1024, 
                                            bar_position=(50, 50),
                                            scale_bar_thickness=20,
                                            font_size=55)
    text_list = ["CA1", "CA3", "Field", "Patch"]  # Example texts
    coordinates_list = [(900, 350), (700, 700), (550, 850), (1125, 600)]  # Example coordinates (x, y)
    image = add_labels_to_image(image, text_list,
                                coordinates_list,font_size=55)

    proj_img = pillow.Image.open(projection_image)#.convert('L')
    if proj_img.mode != "RGB":
        image = proj_img.convert("RGB")


    proj_img= add_scale_bar_with_magnification(image_opened=proj_img, 
                                               magnification=40,
                                               scale_length_um=50,
                                               image_width_px=1024, 
                                               bar_position=(50, 50),
                                               scale_bar_thickness=20,
                                               font_size=55)

    text_list = ["Field"]  # Example texts
    coordinates_list = [(620, 425)]  # Example coordinates (x, y)
    proj_img = add_labels_to_image(proj_img, text_list,
                                   coordinates_list,font_size=55)

    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1, 1, 1
                   ]  # Adjust these values as needed
    height_ratios = [1, 1, 1, 1, 1,
                     1, 1, 1
                    ]       # Adjust these values as needed

    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(8, 10,width_ratios=width_ratios, 
                  height_ratios=height_ratios,figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    axs_img = fig.add_subplot(gs[0:3, 0:3])
    plot_image(image,axs_img,-0.05,0,1.05)
    bpf.add_subplot_label(axs_img, 'A', xpos=0.05, ypos=1.1, fontsize=16, fontweight='bold')
   
    axs_proj = fig.add_subplot(gs[2:5,0:3])
    plot_image(proj_img,axs_proj,-0.045,-0.085,1)
    bpf.add_subplot_label(axs_proj, 'B', xpos=0.05, ypos=1.1, fontsize=16, fontweight='bold')
    
    axs_pat1=fig.add_subplot(gs[0:1,2:4])
    axs_pat2=fig.add_subplot(gs[0:1,4:6])
    axs_pat3=fig.add_subplot(gs[0:1,6:9])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0.05,0)
    move_axis([axs_pat1,axs_pat2],xoffset=0.04,yoffset=0,pltscale=1)

    axs_fl1=fig.add_subplot(gs[3:5,3:5])
    axs_fl2=fig.add_subplot(gs[3:5,5:7])
    axs_fl3=fig.add_subplot(gs[3:5,7:9])
    ylim = (-0.7,0.3) # in mV
    xlim = 150 # in mseconds
    ylabel="field response (mV)"
    plot_trace_raw_all_pats(cell_data,"field_trace(mV)", ylim, xlim,
                            ylabel,axs_fl1,axs_fl2,axs_fl3)
    axs_fl1.set_xlabel(None)
    axs_fl3.set_xlabel(None)
    
    axs_fl2.set_ylabel(None)
    axs_fl3.set_ylabel(None)
    axs_fl2.set_yticklabels([])
    axs_fl3.set_yticklabels([])
    axs_fl_list = [axs_fl1,axs_fl2,axs_fl3]
    fl_labels = bpf.generate_letter_roman_labels("D", len(axs_fl_list))
    bpf.add_subplot_labels_from_list(axs_fl_list, fl_labels, 
                                    base_params={'xpos': -0.1, 'ypos': 1.1, 'fontsize': 16, 'fontweight': 'bold'})
    move_axis(axs_fl_list,xoffset=0,yoffset=-0.025,pltscale=1)

    axs_inset = fig.add_subplot(gs[3:5,2:4])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",
                      axs_inset,0)

    axs_inset = fig.add_subplot(gs[3:5,4:6])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",
                      axs_inset,1)

    axs_inset = fig.add_subplot(gs[3:5,6:8])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",
                      axs_inset,2)

    axs_cl1=fig.add_subplot(gs[1:3,3:5])
    axs_cl2=fig.add_subplot(gs[1:3,5:7])
    axs_cl3=fig.add_subplot(gs[1:3,7:9])
    ylim = (-2,4) # in mV
    xlim = 150 # in mseconds
    ylabel="cell response (mV)"
    plot_trace_raw_all_pats(cell_data,"cell_trace(mV)", ylim, xlim,
                            ylabel,axs_cl1,axs_cl2,axs_cl3)
    axs_cl1.set_xlabel(None)
    axs_cl2.set_xlabel(None)
    axs_cl3.set_xlabel(None)
    axs_cl1.set_xticklabels([])
    axs_cl2.set_xticklabels([])
    axs_cl3.set_xticklabels([])
    axs_cl2.set_ylabel(None)
    axs_cl3.set_ylabel(None)
    axs_cl2.set_yticklabels([])
    axs_cl3.set_yticklabels([])
    axs_cl_list = [axs_cl1,axs_cl2,axs_cl3]
    cl_labels = bpf.generate_letter_roman_labels("C", len(axs_cl_list))
    bpf.add_subplot_labels_from_list(axs_cl_list, cl_labels, 
                                    base_params={'xpos': -0.1, 'ypos': 1.1, 'fontsize': 16, 'fontweight': 'bold'})


    # Fix legend positioning and styling to match the figure
    handles, labels = axs_cl2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Position legend in the top right area to match the figure
    axs_cl3.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor=(1.05, 1),
               ncol=1, title="Voltage trace",
               loc='upper left', frameon=True,
               fancybox=True, shadow=True)    
    
    # Position panel E to match the figure layout (bottom left area)
    axs_cell = fig.add_subplot(gs[6:8,0:2])
    move_axis([axs_cell],xoffset=-0.02,yoffset=0,pltscale=1)

    plot_trace_stats(alltrial_Df,fig,axs_cell)
    bpf.add_subplot_label(axs_cell, 'E', xpos=-0.15, ypos=1.05, fontsize=16, fontweight='bold')

    # Position Fi and Fii panels to match the figure (bottom right area)
    axs_fi = fig.add_subplot(gs[6:8,3:5])   
    axs_fii = fig.add_subplot(gs[6:8,6:8])   
    plot_trace_stats_with_pvalue(alltrial_Df, fig, axs_fi, axs_fii)
    f_axes = [axs_fi, axs_fii]
    f_labels = bpf.generate_letter_roman_labels("F", len(f_axes))
    bpf.add_subplot_labels_from_list(f_axes, f_labels, 
                                    base_params={'xpos': -0.15, 'ypos': 1.05, 'fontsize': 16, 'fontweight': 'bold'})

    plt.tight_layout()
    
    # Using unified saving system - supports multiple formats via command line flags
    bpf.save_figure_smart(fig, outdir, "figure_1")
    
    # Legacy code (commented out - now handled by unified system):
    #outpath = f"{outdir}/figure_1.png"
    #outpath = f"{outdir}/figure_1.svg"
    #outpath = f"{outdir}/figure_1.pdf"
    #plt.savefig(outpath,bbox_inches='tight')
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser with unified plotting system compatibility
    description = '''Generates Figure 1 of the pattern learning paper'''
    parser = argparse.ArgumentParser(description=description)
    
    # Standard arguments (compatible with unified plotting system)
    parser.add_argument('--pikl-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path to the giant pickle file with all cells'
                       )
    parser.add_argument('--illustration-path', '-i'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file in png format'
                       )
    parser.add_argument('--projection-image', '-p'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file showing projections'
                       )
    parser.add_argument('--alltrials-path', '-t'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with all trials'
                        'all cells data in pickle'
                       )
    parser.add_argument('--outdir-path','-o'
                        ,required = False, default ='./', type=str
                        ,help = 'where to save the generated figure image'
                       )
    
    # Additional arguments for unified plotting system compatibility
    # (These are not used by Figure 1 but ensure compatibility)
    parser.add_argument('--sortedcell-path', '-s'
                        , required = False, default=None, type=str
                        , help = 'path to pickle file with cell sorted data (not used by Figure 1)'
                       )
    parser.add_argument('--inR-path', '-r'
                        , required = False, default=None, type=str
                        , help = 'path to pickle file with inR data (not used by Figure 1)'
                       )
    parser.add_argument('--cellstat-path', '-c'
                        , required = False, default=None, type=str
                        , help = 'path to cell statistics file (not used by Figure 1)'
                       )
    parser.add_argument('--training-path', '-n'
                        , required = False, default=None, type=str
                        , help = 'path to training data file (not used by Figure 1)'
                       )
    parser.add_argument('--firingproperties-path', '-q'
                        , required = False, default=None, type=str
                        , help = 'path to firing properties file (not used by Figure 1)'
                       )
    parser.add_argument('--patillustration-path', '-m'
                        , required = False, default=None, type=str
                        , help = 'path to pattern illustration file (not used by Figure 1)'
                       )
    #    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    pklpath = Path(args.pikl_path)
    illustration_path = Path(args.illustration_path)
    projection_path = Path(args.projection_image)
    all_trials_path= Path(args.alltrials_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_1'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_1(pklpath, all_trials_path, illustration_path,
                  projection_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
