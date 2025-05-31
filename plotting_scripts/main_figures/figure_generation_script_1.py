__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 1 of pattern learning paper.
Takes in the pickle file that stores all the experimental data.
Takes in the image files with slice and pipettes showing recording location and
the fluroscence on CA3.
Generates the plot showing the size of the grids/points in patterns.
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
import sys
import os

# Add the src directory to the path to import our shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from shared_utilities import (PatternLearningUtils, set_plot_properties, create_grid_image, 
                            subtract_baseline, convert_pvalue_to_asterisks, pre_color, 
                            post_color, post_late, CB_color_cycle, color_fader)

from PIL import ImageDraw, ImageFont 
from scipy.stats import gaussian_kde
from scipy.stats import levene

# Initialize utilities
utils = PatternLearningUtils()

# plot features are defined in shared utilities
set_plot_properties()

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
            pat_fr = create_grid_image(0,1.45)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = create_grid_image(4,1.45)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = create_grid_image(17,1.45)
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
            trace = subtract_baseline(trace,bl_period_in_ms=2)
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
                trace = subtract_baseline(trace,bl_period_in_ms=2)
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


def plot_trace_stats(feature_extracted_df, fig, axs_cell, axs_field): 
                     #axs_cell_t, axs_field_t):
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
    
    # Plot bar distributions using the updated bar distribution function
    max_count_cell, bins_cell = plot_bar_distribution(
        axs_cell, 
        merged_df['max_trace'], 
        color='black', 
        label='EPSP', 
        show_ylabel=True
    )
    
    max_count_field, bins_field = plot_bar_distribution(
        axs_field, 
        merged_df['min_field'], 
        color='black', 
        label='LFP', 
        show_ylabel=False, 
        remove_ytick_labels=True
    )
    
    #max_count_cell_t, bins_cell_t = plot_bar_distribution(
    #    axs_cell_t, 
    #    merged_df['max_trace_t'], 
    #    color='black', 
    #    label='EPSP (t)', 
    #    show_ylabel=True
    #)
    #
    #max_count_field_t, bins_field_t = plot_bar_distribution(
    #    axs_field_t, 
    #    merged_df['min_field_t'], 
    #    color='black', 
    #    label='LFP (t)', 
    #    show_ylabel=False, 
    #    remove_ytick_labels=True
    #)
    
    if bins_cell is not None and bins_field is not None:
        combined_xlim_cell = (min(bins_cell.min(), bins_field.min()), max(bins_cell.max(), bins_field.max()))
        axs_cell.set_xlim(combined_xlim_cell)
        axs_field.set_xlim(combined_xlim_cell)
    
    #if bins_cell_t is not None and bins_field_t is not None:
    #    combined_xlim_cell_t = (min(bins_cell_t.min(), bins_field_t.min()), max(bins_cell_t.max(), bins_field_t.max()))
    #    axs_cell_t.set_xlim(combined_xlim_cell_t)
    #    axs_field_t.set_xlim(combined_xlim_cell_t)

    max_count_1 = max(max_count_cell, max_count_field)
    axs_cell.set_ylim(0, max_count_1 * 1.01)
    axs_field.set_ylim(0, max_count_1 * 1.01)
    
    #max_count_2 = max(max_count_cell_t, max_count_field_t)
    #axs_cell_t.set_ylim(0, max_count_2 * 1.01)
    #axs_field_t.set_ylim(0, max_count_2 * 1.01)

##everything works 
#def plot_trace_stats(feature_extracted_df, fig, axs_cell, axs_field,
#                     axs_cell_t, axs_field_t):
#    """
#    Plot bar distributions of `max_trace`, `min_field`, `max_trace_t`, and `min_field_t`
#    after normalizing their mean using the combined mean of trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`.
#    
#    Parameters:
#    - feature_extracted_df (pd.DataFrame): Input DataFrame containing features.
#    - fig (plt.Figure): Figure object.
#    - axs_cell (plt.Axes): Axis for `max_trace` plot.
#    - axs_field (plt.Axes): Axis for `min_field` plot.
#    - axs_cell_t (plt.Axes): Axis for `max_trace_t` plot.
#    - axs_field_t (plt.Axes): Axis for `min_field_t` plot.
#    """
#    
#    # Check if required columns exist in the DataFrame
#    required_cols = ['pre_post_status', 'trial_no', 'cell_ID', 'frame_id', 
#                     'max_trace', 'min_field', 'max_trace_t', 'min_field_t']
#    if not all(col in feature_extracted_df.columns for col in required_cols):
#        raise ValueError("Input DataFrame is missing required columns.")
#    
#    # Step 1: Filter data for 'pre' status
#    filtered_df = feature_extracted_df[feature_extracted_df['pre_post_status'] == 'pre']
#    
#    # Step 2: Calculate the mean across trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`
#    mean_combined_df = filtered_df[
#        filtered_df['trial_no'].isin([0, 1, 2])
#    ].groupby(['cell_ID', 'frame_id'])[['max_trace', 'min_field', 'max_trace_t', 'min_field_t']].mean().reset_index()
#    
#    # Step 3: Merge the combined mean back to the original filtered data
#    merged_df = filtered_df.merge(mean_combined_df, on=['cell_ID', 'frame_id'], suffixes=('', '_mean'))
#    
#    # Step 4: Normalize the columns using the combined mean
#    merged_df['max_trace'] = merged_df['max_trace'] / merged_df['max_trace_mean']
#    merged_df['min_field'] = merged_df['min_field'] / merged_df['min_field_mean']
#    merged_df['max_trace_t'] = merged_df['max_trace_t'] / merged_df['max_trace_t_mean']
#    merged_df['min_field_t'] = merged_df['min_field_t'] / merged_df['min_field_t_mean']
#    
#    # Helper function to plot bar distributions without y-axis normalization
#    def plot_bar_distribution(ax, data, color, label, title, show_ylabel=True, remove_ytick_labels=False):
#        if len(data) > 1:
#            # Plot the histogram as a bar plot (without normalizing the counts)
#            n, bins, patches = ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor='black')
#            ax.set_xlabel(f'Normalized\n{label}')
#            
#            if show_ylabel:
#                ax.set_ylabel('Count')
#            if remove_ytick_labels:
#                ax.set_yticklabels([])  # Remove only the y-axis tick labels
#            
#            ax.spines['top'].set_visible(False)
#            ax.spines['right'].set_visible(False)
#            return n.max(), bins  # Return the maximum count and bins for x-axis scaling
#        else:
#            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
#            return 0, None
#
#    # Step 5: Plot bar distributions for each feature on their respective axes
#    max_count_cell, bins_cell = plot_bar_distribution(
#        axs_cell, 
#        merged_df['max_trace'], 
#        color='black', 
#        label='EPSP', 
#        title='EPSP', 
#        show_ylabel=True
#    )
#    
#    max_count_field, bins_field = plot_bar_distribution(
#        axs_field, 
#        merged_df['min_field'], 
#        color='black', 
#        label='LFP', 
#        title='LFP', 
#        show_ylabel=False, 
#        remove_ytick_labels=True
#    )
#    
#    max_count_cell_t, bins_cell_t = plot_bar_distribution(
#        axs_cell_t, 
#        merged_df['max_trace_t'], 
#        color='black', 
#        label='EPSP (t)', 
#        title='EPSP (t)', 
#        show_ylabel=True
#    )
#    
#    max_count_field_t, bins_field_t = plot_bar_distribution(
#        axs_field_t, 
#        merged_df['min_field_t'], 
#        color='black', 
#        label='LFP (t)', 
#        title='LFP (t)', 
#        show_ylabel=False, 
#        remove_ytick_labels=True
#    )
#    
#    # Step 6: Set the same x-axis limits for `max_trace` and `min_field`
#    if bins_cell is not None and bins_field is not None:
#        combined_xlim_cell = (
#            min(bins_cell.min(), bins_field.min()), 
#            max(bins_cell.max(), bins_field.max())
#        )
#        axs_cell.set_xlim(combined_xlim_cell)
#        axs_field.set_xlim(combined_xlim_cell)
#    
#    ## Step 7: Set the same x-axis limits for `max_trace_t` and `min_field_t`
#    if bins_cell_t is not None and bins_field_t is not None:
#        combined_xlim_cell_t = (
#            min(bins_cell_t.min(), bins_field_t.min()), 
#            max(bins_cell_t.max(), bins_field_t.max())
#        )
#        axs_cell_t.set_xlim(combined_xlim_cell_t)
#        axs_field_t.set_xlim(combined_xlim_cell_t)
#    
#    # Step 8: Set the same y-axis limits for each pair of plots
#    max_count_1 = max(max_count_cell, max_count_field)
#    axs_cell.set_ylim(0, max_count_1 * 1.01)
#    axs_field.set_ylim(0, max_count_1 * 1.01)
#    
#    max_count_2 = max(max_count_cell_t, max_count_field_t)
#    axs_cell_t.set_ylim(0, max_count_2 * 1.01)
#    axs_field_t.set_ylim(0, max_count_2 * 1.01)


##everything works but plots only EPSP and field amplitudes
#def plot_trace_stats(feature_extracted_df, fig, axs_cell, axs_field):
#    """
#    Plot bar distributions of `max_trace` and `min_field`
#    after normalizing their mean using the combined mean of trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`.
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
#    # Step 2: Calculate the mean across trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`
#    mean_combined_df = filtered_df[
#        filtered_df['trial_no'].isin([0, 1, 2])
#    ].groupby(['cell_ID', 'frame_id'])[['max_trace', 'min_field']].mean().reset_index()
#    
#    # Step 3: Merge the combined mean back to the original filtered data
#    merged_df = filtered_df.merge(mean_combined_df, on=['cell_ID', 'frame_id'], suffixes=('', '_mean'))
#    
#    # Step 4: Normalize `max_trace` and `min_field` using the combined mean
#    merged_df['max_trace'] = merged_df['max_trace'] / merged_df['max_trace_mean']
#    merged_df['min_field'] = merged_df['min_field'] / merged_df['min_field_mean']
#    
#    # Helper function to plot bar distributions without y-axis normalization
#    def plot_bar_distribution(ax, data, color, label, title, show_ylabel=True, remove_ytick_labels=False):
#        if len(data) > 1:
#            # Plot the histogram as a bar plot (without normalizing the counts)
#            n, bins, patches = ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor='black')
#            ax.set_xlabel(f'Normalized\n{label}')
#            
#            if show_ylabel:
#                ax.set_ylabel('Count')
#            if remove_ytick_labels:
#                ax.set_yticklabels([])  # Remove only the y-axis tick labels
#            
#            ax.spines['top'].set_visible(False)
#            ax.spines['right'].set_visible(False)
#            #ax.set_title(title)
#            
#            return n.max()  # Return the maximum count for y-axis scaling
#        else:
#            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
#            return 0
#
#    # Step 5: Plot bar distribution for `max_trace` on axs_cell with title "EPSP"
#    max_count_cell = plot_bar_distribution(
#        axs_cell, 
#        merged_df['max_trace'], 
#        color='black', 
#        label='EPSP', 
#        title='EPSP', 
#        show_ylabel=True
#    )
#    
#    # Step 6: Plot bar distribution for `min_field` on axs_field with title "LFP" and no y-tick labels
#    max_count_field = plot_bar_distribution(
#        axs_field, 
#        merged_df['min_field'], 
#        color='black', 
#        label='LFP', 
#        title='LFP', 
#        show_ylabel=False, 
#        remove_ytick_labels=True
#    )
#    
#    # Step 7: Set the same x-axis limits for both plots
#    combined_xlim = (
#        min(axs_cell.get_xlim()[0], axs_field.get_xlim()[0]), 
#        max(axs_cell.get_xlim()[1], axs_field.get_xlim()[1])
#    )
#    axs_cell.set_xlim(combined_xlim)
#    axs_field.set_xlim(combined_xlim)
#    
#    # Step 8: Set the same y-axis limits for both plots
#    max_count = max(max_count_cell, max_count_field)
#    axs_cell.set_ylim(0, max_count * 1.01)
#    axs_field.set_ylim(0, max_count * 1.01)


def plot_trace_stats_with_pvalue(feature_extracted_df, fig, ax):
    """
    Plot bar distributions of `max_trace` and `min_field` on a single axis,
    and display the p-value from Levene's test for equality of variances.
    
    Parameters:
    - feature_extracted_df (pd.DataFrame): Input DataFrame containing features.
    - fig (plt.Figure): Figure object.
    - ax (plt.Axes): Axis for the combined plot.
    """
    
    # Check if required columns exist in the DataFrame
    required_cols = ['pre_post_status', 'trial_no', 'cell_ID', 'frame_id', 'max_trace', 'min_field']
    if not all(col in feature_extracted_df.columns for col in required_cols):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Step 1: Filter data for 'pre' status
    filtered_df = feature_extracted_df[feature_extracted_df['pre_post_status'] == 'pre']
    
    # Step 2: Calculate the mean across trials 0, 1, 2 for each grouped `cell_ID` and `frame_id`
    mean_combined_df = filtered_df[
        filtered_df['trial_no'].isin([0, 1, 2])
    ].groupby(['cell_ID', 'frame_id'])[['max_trace', 'min_field']].mean().reset_index()
    
    # Step 3: Merge the combined mean back to the original filtered data
    merged_df = filtered_df.merge(mean_combined_df, on=['cell_ID', 'frame_id'], suffixes=('', '_mean'))
    
    # Step 4: Normalize `max_trace` and `min_field` using the combined mean
    merged_df['max_trace'] = merged_df['max_trace'] / merged_df['max_trace_mean']
    merged_df['min_field'] = merged_df['min_field'] / merged_df['min_field_mean']
    
    # Step 5: Create a DataFrame for plotting
    data_to_plot = pd.DataFrame({
        'Normalised Mean': np.concatenate([merged_df['max_trace'].dropna(), merged_df['min_field'].dropna()]),
        'Category': ['EPSP'] * len(merged_df['max_trace'].dropna()) + ['LFP'] * len(merged_df['min_field'].dropna())
    })
    
    # Step 6: Perform Levene's test to compare variances
    stat, p_value = levene(merged_df['max_trace'].dropna(), merged_df['min_field'].dropna())
    pval_list = [p_value]  # Create a list for annotation
    
    # Convert p-value to asterisks using your custom function
    annotations = [convert_pvalue_to_asterisks(p) for p in pval_list]
    
    # Step 7: Create the box plot on a single axis
    sns.boxplot(
        data=data_to_plot, 
        x='Category', 
        y='Normalised Mean', 
        ax=ax, 
        boxprops=dict(facecolor='grey', edgecolor='darkgrey'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='darkgrey'),
        capprops=dict(color='darkgrey'),
        flierprops=dict(markerfacecolor='black', marker='o', 
                        markersize=3, linestyle='none',alpha=0.4)
    )
    
    # Remove right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add y-axis label
    ax.set_ylabel('Normalised\nmean')
    
    # Set x-axis label to "Variance" and remove the title
    ax.set_xlabel('Variance')
    ax.set_xticklabels(['EPSP', 'LFP'])
    
    # Step 8: Use `statannotations.Annotator` for annotation
    pairs = [('EPSP', 'LFP')]  # Specify the pairs to annotate
    annotator = Annotator(ax, pairs, data=data_to_plot, x='Category', y='Normalised Mean')
    annotator.set_custom_annotations(annotations)
    annotator.annotate()
    
    # Adjust y-limits for better visibility of annotations
    ax.set_ylim(0, data_to_plot['Normalised Mean'].max() * 1.2)

def main():
    """Main function using shared utilities system"""
    description = '''Generates figure 1'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, default='.', 
                       help='Base data directory')
    parser.add_argument('--analysis-type', type=str, default='standard',
                       choices=['standard', 'field_normalized'],
                       help='Analysis type')
    args = parser.parse_args()

    # Initialize utilities
    utils = PatternLearningUtils(config_path=os.path.join(args.data_dir, 'config.yaml'))
    
    try:
        # Load figure data using the utilities system
        figure_data = utils.load_figure_data('figure_1', args.analysis_type)
        
        # Extract data components
        baseline_traces = figure_data['baseline_traces_all_cells']
        with_fluorescence_pipette = figure_data['with_fluorescence_pipette']
        screenshot_2023 = figure_data['screenshot_2023']
        pd_all_cells_all_trials = figure_data['pd_all_cells_all_trials']
        
        # Generate figure
        fig = plot_figure_1_new(baseline_traces, with_fluorescence_pipette,
                               screenshot_2023, pd_all_cells_all_trials,
                               cell_to_plot)
        
        # Save figure using standardized output manager
        saved_files = utils.output_manager.save_figure(
            fig, 'figure_1', 'main_figures', args.analysis_type
        )
        
        utils.logger.info(f"Figure 1 generated successfully: {saved_files}")
        
        plt.close(fig)
        
    except Exception as e:
        utils.logger.error(f"Error generating Figure 1: {e}")
        raise


def plot_figure_1_new(baseline_traces, with_fluorescence_pipette, screenshot_2023, 
                     pd_all_cells_all_trials, cell_to_plot):
    """Generate Figure 1 using the loaded data with standardized utilities"""
    set_plot_properties()
    
    # Get specific cell data
    cell_data = pd_all_cells_all_trials[pd_all_cells_all_trials["cell_ID"] == cell_to_plot]
    
    # Process images
    image = with_fluorescence_pipette.convert("RGB")
    proj_img = screenshot_2023.convert("RGB")
    
    proj_img = add_scale_bar_with_magnification(
        image_opened=proj_img, 
        magnification=40,
        scale_length_um=50,
        image_width_px=1024, 
        bar_position=(50, 50),
        scale_bar_thickness=20,
        font_size=55
    )

    text_list = ["Field"]
    coordinates_list = [(620, 425)]
    proj_img = add_labels_to_image(proj_img, text_list, coordinates_list, font_size=55)

    # Create figure with standardized layout
    width_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    height_ratios = [1, 1, 1, 1, 1, 1, 1, 1]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(8, 10, width_ratios=width_ratios, 
                  height_ratios=height_ratios, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    # Plot images
    axs_img = fig.add_subplot(gs[0:3, 0:3])
    plot_image(image, axs_img, -0.05, 0, 1.05)
    axs_img.text(0.05, 1.1, 'A', transform=axs_img.transAxes,    
                fontsize=16, fontweight='bold', ha='center', va='center')
   
    axs_proj = fig.add_subplot(gs[2:5, 0:3])
    plot_image(proj_img, axs_proj, -0.045, -0.085, 1)
    axs_proj.text(0.05, 1.1, 'B', transform=axs_proj.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Plot patterns
    axs_pat1 = fig.add_subplot(gs[0:1, 2:4])
    axs_pat2 = fig.add_subplot(gs[0:1, 4:6])
    axs_pat3 = fig.add_subplot(gs[0:1, 6:9])
    plot_patterns(axs_pat1, axs_pat2, axs_pat3, 0.05, 0)
    move_axis([axs_pat1, axs_pat2], xoffset=0.04, yoffset=0, pltscale=1)

    # Plot field traces
    axs_fl1 = fig.add_subplot(gs[3:5, 3:5])
    axs_fl2 = fig.add_subplot(gs[3:5, 5:7])
    axs_fl3 = fig.add_subplot(gs[3:5, 7:9])
    ylim = (-0.7, 0.3)
    xlim = 150
    ylabel = "field response (mV)"
    plot_trace_raw_all_pats(cell_data, "field_trace(mV)", ylim, xlim,
                           ylabel, axs_fl1, axs_fl2, axs_fl3)
    
    # Format field trace axes
    axs_fl1.set_xlabel(None)
    axs_fl3.set_xlabel(None)
    axs_fl2.set_ylabel(None)
    axs_fl3.set_ylabel(None)
    axs_fl2.set_yticklabels([])
    axs_fl3.set_yticklabels([])
    axs_fl_list = [axs_fl1, axs_fl2, axs_fl3]
    label_axis(axs_fl_list, "D", xpos=-0.1, ypos=1.1)
    move_axis(axs_fl_list, xoffset=0, yoffset=-0.025, pltscale=1)

    # Plot inset traces
    for i, ax_pos in enumerate([(2, 4), (4, 6), (6, 8)]):
        axs_inset = fig.add_subplot(gs[3:5, ax_pos[0]:ax_pos[1]])
        inset_plot_traces(cell_data, "field_trace(mV)", axs_inset, i)

    # Plot cell traces
    axs_cl1 = fig.add_subplot(gs[1:3, 3:5])
    axs_cl2 = fig.add_subplot(gs[1:3, 5:7])
    axs_cl3 = fig.add_subplot(gs[1:3, 7:9])
    ylim = (-2, 4)
    xlim = 150
    ylabel = "cell response (mV)"
    plot_trace_raw_all_pats(cell_data, "cell_trace(mV)", ylim, xlim,
                           ylabel, axs_cl1, axs_cl2, axs_cl3)
    
    # Format cell trace axes
    for ax in [axs_cl1, axs_cl2, axs_cl3]:
        ax.set_xlabel(None)
        ax.set_xticklabels([])
    
    axs_cl2.set_ylabel(None)
    axs_cl3.set_ylabel(None)
    axs_cl2.set_yticklabels([])
    axs_cl3.set_yticklabels([])
    axs_cl_list = [axs_cl1, axs_cl2, axs_cl3]
    label_axis(axs_cl_list, "C", xpos=-0.1, ypos=1.1)

    # Add legend
    handles, labels = axs_cl2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs_cl2.legend(by_label.values(), by_label.keys(), 
                  bbox_to_anchor=(0.8, 1),
                  ncol=1, title="Voltage trace",
                  loc='upper center', frameon=False)
    
    # Plot statistics
    axs_cell = fig.add_subplot(gs[6:9, 1:3])
    move_axis([axs_cell], xoffset=-0.05, yoffset=0, pltscale=1)
    axs_field = fig.add_subplot(gs[6:9, 3:5])
    move_axis([axs_field], xoffset=0.015, yoffset=0, pltscale=1)

    plot_trace_stats(pd_all_cells_all_trials, fig, axs_cell, axs_field)
    label_axis([axs_cell, axs_field], "E", xpos=-0.1, ypos=1)

    axs_stat_dist = fig.add_subplot(gs[6:9, 6:8])   
    plot_trace_stats_with_pvalue(pd_all_cells_all_trials, fig, axs_stat_dist)
    axs_stat_dist.text(-0.2, 1, 'F', transform=axs_stat_dist.transAxes,    
                      fontsize=16, fontweight='bold', ha='center', va='center')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import time
    ts = time.time()
    main()
    tf = time.time()
    print(f'Total time = {np.around(((tf-ts)/60), 1)} (mins)')
