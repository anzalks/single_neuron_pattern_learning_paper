__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 5 of pattern learning paper.
Takes in the pickle file that stores all the experimental data.
Takes in the image files with slice and pipettes showing recordin location and
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
from matplotlib.transforms import Affine2D
import sys
import os

# Add the src directory to the path to import our shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from shared_utilities import (PatternLearningUtils, set_plot_properties, create_grid_image, 
                            subtract_baseline, convert_pvalue_to_asterisks, pre_color, 
                            post_color, post_late, CB_color_cycle, color_fader)


# Initialize utilities
utils = PatternLearningUtils()

# plot features are defines in bpf
set_plot_properties()

vlinec = "#C35817"

time_to_plot = 0.250 # in s 

time_points = ["pre","0", "10", "20","30" ]
selected_time_points = ['post_0', 'post_1', 'post_2', 'post_3','pre']
                        #'post_4','post_5']

class Args: pass
args_ = Args()

def plot_patterns(axs_pat1,axs_pat2,axs_pat3,xoffset,yoffset,title_row_num):
    if title_row_num==1:
        pattern_list = ["trained pattern","Overlapping pattern",
                        "Non-overlapping pattern"]
    else:
        pattern_list = ["trained\npattern","Overlapping\npattern",
                        "Non-overlapping\npattern"]

    for pr_no, pattern in enumerate(pattern_list):
        if pr_no==0:
            axs_pat = axs_pat1  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = create_grid_image(0,2)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = create_grid_image(4,2)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = create_grid_image(17,2)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        pat_pos = axs_pat.get_position()
        new_pat_pos = [pat_pos.x0+xoffset, pat_pos.y0+yoffset, pat_pos.width,
                        pat_pos.height]
        axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        axs_pat.set_title(pattern,fontsize=10)
    
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
        "X", "IX", "V", "IV",
        "I"
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


def normalise_df_to_pre(all_trial_cell_df,field_to_plot):
    cell_grp=all_trial_cell_df.groupby(by="cell_ID")
    c_df = all_trial_cell_df.copy()
    for cell, cell_data in cell_grp:
        pps_grp = cell_data.groupby(by="pre_post_status")
        for pps,pps_data in pps_grp:
            if pps=="pre":
                pass
            elif pps=="post_3":
                pass
            else:
                pass
            trial_grp = pps_data.groupby(by="trial_no")
            for trial_no,trial_data in trial_grp:
                frame_grp = trial_data.groupby(by="frame_id")
                for frame, frame_data in frame_grp:
                    pre_data =c_df[(c_df["cell_ID"]==cell)&(c_df["pre_post_status"]=="pre")&(c_df["trial_no"]==trial_no)&(c_df["frame_id"]==frame)][field_to_plot].to_numpy()
                    norm_data =(frame_data[field_to_plot].to_numpy()/pre_data)*100
                    if norm_data==0:
                        norm_data=np.nan
                        print(f"pre_data: {pre_data},zero is there")
                    else:
                        norm_data=norm_data
                        print(f"pre_data, {pre_data} no zero")
                    #print(f"pps:{pps}, pre: {pre_data}" )
                    c_df[field_to_plot].iloc[(c_df["cell_ID"]==cell)&(c_df["pre_post_status"]==pps)&(c_df["trial_no"]==trial_no)&(c_df["frame_id"]==frame)]=norm_data
    return c_df

def plot_mini_feature(cells_df,field_to_plot, learners,non_learners,fig,axs):
    if field_to_plot=="mepsp_amp":
        ylim=(0,1.3)
    elif field_to_plot=="freq_mepsp":
        ylim=(0,10)
    elif field_to_plot=="num_mepsp":
        ylim=(0,50)
    elif field_to_plot=="mepsp_time":
        ylim=(0,100)
    else:
        ylim=(None,None)
    
    # Check if the required column exists
    if field_to_plot not in cells_df.columns:
        raise ValueError(f"Required column '{field_to_plot}' not found in data. "
                        f"Available columns: {list(cells_df.columns)}")
    
    order = np.array(["pre","post_3"])
    cells_df=cells_df.copy()
    data_to_plot = cells_df[cells_df["pre_post_status"].isin(["pre","post_3"])]
    learners_df = data_to_plot[data_to_plot["cell_ID"].isin(learners)]
    non_learners_df = data_to_plot[data_to_plot["cell_ID"].isin(non_learners)]
    pre_dat = learners_df[learners_df["pre_post_status"]=="pre"][field_to_plot]
    post_dat = learners_df[learners_df["pre_post_status"]=="post_3"][field_to_plot]

    sns.pointplot(data=learners_df,x="pre_post_status",y=field_to_plot,ax=axs,
                 order=order,color=CB_color_cycle[0],
                 errorbar='sd')
    sns.pointplot(data=non_learners_df,x="pre_post_status",
                  y=field_to_plot,ax=axs,order=order,
                  color=CB_color_cycle[1],errorbar='sd')
    stat_analysis= spst.wilcoxon(pre_dat,post_dat, zero_method="wilcox", correction=True)
    pvalList = stat_analysis.pvalue
    anotp_list=["pre","post_3"]
    annotator =Annotator(axs,[anotp_list], data=learners_df, 
                         x="pre_post_status",y=field_to_plot,order=order)
    annotator.set_custom_annotations([convert_pvalue_to_asterisks(pvalList)])
    #annotator.annotate()
    axs.set_title(field_to_plot)
    axs.set_ylabel(None)
    axs.set_ylim(ylim)
    return None
   
def plot_learner_vs_non_learner_mini_feature(cells_df,field_to_plot,learners,non_learners,fig,axs):
    if field_to_plot=="mepsp_amp":
        ylim=(0,1.3)
    elif field_to_plot=="freq_mepsp":
        ylim=(0,10)
    elif field_to_plot=="num_mepsp":
        ylim=(0,50)
    elif field_to_plot=="mepsp_time":
        ylim=(0,100)
    else:
        ylim=(None,None)
    
    # Check if the required column exists
    if field_to_plot not in cells_df.columns:
        raise ValueError(f"Required column '{field_to_plot}' not found in data. "
                        f"Available columns: {list(cells_df.columns)}")
    
    order = np.array(["learners","non_learners"])
    cells_df=cells_df.copy()
    data_to_plot = cells_df[cells_df["pre_post_status"].isin(["pre","post_3"])]
    learners_df = data_to_plot[data_to_plot["cell_ID"].isin(learners)]
    non_learners_df = data_to_plot[data_to_plot["cell_ID"].isin(non_learners)]
    learners_dat = learners_df[learners_df["pre_post_status"]=="post_3"][field_to_plot].to_numpy()
    non_learners_dat = non_learners_df[non_learners_df["pre_post_status"]=="post_3"][field_to_plot].to_numpy()
    len_learners = len(learners_dat)
    len_non_learners = len(non_learners_dat)
    if len_learners > len_non_learners:
        non_learners_dat = np.pad(non_learners_dat, 
                                  (0, len_learners -len_non_learners),
                                  constant_values=np.nan)
    else:
        learners_dat = np.pad(learners_dat, 
                              (0, len_non_learners -len_learners),
                              constant_values=np.nan)
    lrn_df = pd.DataFrame({"learners":learners_dat,"non_learners":non_learners_dat})
    long_df = lrn_df.melt(var_name='Category', value_name='Values')

    sns.pointplot(data=long_df, x='Category', y='Values',ax=axs,
                 color=CB_color_cycle[2],errorbar='sd')
    stat_analysis= spst.f_oneway(learners_dat,non_learners_dat)
    pvalList = stat_analysis.pvalue
    anotp_list=["learners","non_learners"]
    annotator =Annotator(axs,[anotp_list], data=long_df,
                         x="Category",y="Values",order=order)
    annotator.set_custom_annotations([convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()
    axs.set_title(field_to_plot)
    axs.set_ylabel(None)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
    axs.set_ylim(ylim)
    return None





def plot_mini_distribution(df_cells,dict_cell_classified,
                           fig, axs1,axs2,axs3,axs4):
    learners = dict_cell_classified["ap_cells"]["cell_ID"].unique()
    non_learners = dict_cell_classified["an_cells"]["cell_ID"].unique()
    
    # Use original mepsp columns - will fail if data is missing
    try:
        plot_mini_feature(df_cells,"mepsp_amp",learners,non_learners,fig,axs1)
        plot_mini_feature(df_cells,"freq_mepsp",learners,non_learners,fig,axs2)
        plot_learner_vs_non_learner_mini_feature(df_cells,"mepsp_amp",
                                                learners,non_learners,fig,axs3)
        plot_learner_vs_non_learner_mini_feature(df_cells,"freq_mepsp",
                                                 learners,non_learners,fig,axs4)
    except ValueError as e:
        print(f"\n{'='*80}")
        print("FIGURE 8 DATA ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print("\nFigure 8 requires mepsp (miniature EPSP) data columns:")
        print("  - mepsp_amp: miniature EPSP amplitude")
        print("  - freq_mepsp: frequency of miniature EPSPs")
        print("  - num_mepsp: number of miniature EPSPs")
        print("  - mepsp_time: timing of miniature EPSPs")
        print("\nTo fix this, regenerate the pickle files with mepsp data:")
        print("python analysis_scripts/extract_features_and_save_pickle.py \\")
        print("    -f /path/to/all_data_with_training_df.pickle \\")
        print("    -s /path/to/cell_stats.h5 \\")
        print("    -o analysis_scripts/pickle_files_from_analysis/")
        print(f"{'='*80}")
        raise

def plot_figure_8(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  cell_stats_pickle_file,
                  outdir
                 ):
    all_cell_all_trial_df = pd.read_pickle(extracted_feature_pickle_file_path)
    sc_data_dict = pd.read_pickle(cell_categorised_pickle_file)
    
    # Define the width and height ratios
    height_ratios = [1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1,
                    ]
                     # Adjust these values as needed
    
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1
                   ]# Adjust these values as needed

    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.5, hspace=0.5)



    #plot distribution of minis
    axs_mini_amp = fig.add_subplot(gs[1:3,0:1])
    axs_mini_freq = fig.add_subplot(gs[1:3,1:2])
    axs_mini_comp_amp = fig.add_subplot(gs[1:3,2:3])
    axs_mini_comp_freq = fig.add_subplot(gs[1:3,3:4])
    plot_mini_distribution(all_cell_all_trial_df,sc_data_dict, fig, 
                           axs_mini_amp,axs_mini_freq,
                           axs_mini_comp_amp,axs_mini_comp_freq)
    axs_mini_list = [axs_mini_amp,axs_mini_freq,
                     axs_mini_comp_amp,axs_mini_comp_freq]
    #label_axis(axs_mini_list,"A")

    plt.tight_layout()
    outpath = f"{outdir}/figure_8.png"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    """Main function using shared utilities system"""
    description = '''Generates figure 8'''
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
        figure_data = utils.load_figure_data('figure_8', args.analysis_type)
        
        # Extract data components
        pd_all_cells_mean = figure_data['pd_all_cells_mean']
        all_cells_classified_dict = figure_data['all_cells_classified_dict']
        
        # Generate figure using the loaded data
        fig = plot_figure_8_adapted(pd_all_cells_mean, all_cells_classified_dict, args.analysis_type)
        
        # Save the figure
        output_files = utils.output_manager.save_figure(
            fig, 'figure_8', 'main_figures', args.analysis_type
        )
        utils.logger.info(f"Figure 8 generated successfully: {output_files}")
        
    except Exception as e:
        utils.logger.error(f"Error generating Figure 8: {e}")
        raise
    finally:
        plt.close('all')


def plot_figure_8_adapted(pd_all_cells_mean, all_cells_classified_dict, analysis_type):
    """Adapted version of plot_figure_8 that works with data objects instead of file paths"""
    
    # Use the data objects directly instead of loading from files
    all_cell_all_trial_df = pd_all_cells_mean  # DataFrame instead of path
    sc_data_dict = all_cells_classified_dict  # Dict instead of path
    
    # Define the width and height ratios
    height_ratios = [1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1,
                    ]
                     # Adjust these values as needed
    
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1
                   ]# Adjust these values as needed

    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.5, hspace=0.5)

    #plot distribution of minis
    axs_mini_amp = fig.add_subplot(gs[1:3,0:1])
    axs_mini_freq = fig.add_subplot(gs[1:3,1:2])
    axs_mini_comp_amp = fig.add_subplot(gs[1:3,2:3])
    axs_mini_comp_freq = fig.add_subplot(gs[1:3,3:4])
    plot_mini_distribution(all_cell_all_trial_df,sc_data_dict, fig, 
                           axs_mini_amp,axs_mini_freq,
                           axs_mini_comp_amp,axs_mini_comp_freq)
    axs_mini_list = [axs_mini_amp,axs_mini_freq,
                     axs_mini_comp_amp,axs_mini_comp_freq]
    #label_axis(axs_mini_list,"A")

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Timing the run with time.time
    ts = time.time()
    main()
    tf = time.time()
    print(f'Total time = {np.around(((tf-ts)/60),1)} (mins)')
