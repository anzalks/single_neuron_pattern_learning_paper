__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Figure 5: Temporal Dynamics

This script generates Figure 5 of the pattern learning paper, which shows:
- Temporal dynamics of synaptic plasticity during pattern learning
- Time-course analysis of EPSP amplitude changes
- Pattern-specific temporal evolution of responses
- Detailed analysis of learning time constants and dynamics
- Comparison of temporal properties between learners and non-learners
- Statistical analysis of temporal plasticity mechanisms

Input files:
- pd_all_cells_mean.pickle: Mean cellular responses
- all_cells_classified_dict.pickle: Cell classification data
- pd_all_cells_all_trials.pickle: Trial-by-trial temporal data
- cell_stats.h5: Cell statistics and temporal properties
- Figure_5_1.png: Illustration of temporal analysis setup

Output: Figure_5/figure_5.png showing comprehensive temporal dynamics analysis
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
from shared_utils import baisic_plot_fuctnions_and_features as bpf
import re
from scipy.stats import ttest_1samp
from scipy.stats import spearmanr
from PIL import ImageDraw, ImageFont

# plot features are defines in bpf
bpf.set_plot_properties()

vlinec = "#C35817"

learner_cell = "2022_12_21_cell_1" 
non_learner_cell = "2023_01_10_cell_1"
time_to_plot = 0.250 # in s 

time_points = ["pre","0", "10", "20","30" ]
selected_time_points = ['post_0', 'post_1', 'post_2', 'post_3','pre']
                        #'post_4','post_5']
cell_dist=[8,10,4]
cell_dist_key = ["leaners","non\nlearners","cells\nnot\ncosidered"]

class Args: pass
args_ = Args()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def human_sort(lst):
    return sorted(lst, key=natural_sort_key)
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
            pat_fr = bpf.create_grid_image(0,1.5)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,1.5)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,1.5)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        pat_pos = axs_pat.get_position()
        new_pat_pos = [pat_pos.x0+xoffset, pat_pos.y0+yoffset, pat_pos.width,
                        pat_pos.height]
        axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        axs_pat.set_title(pattern,fontsize=10)

def plot_points(axs_points_img,xoffset,yoffset,zoom):
    first_spot_grid_points = [1, 3, 5, 7, 9, 
                              11, 13, 
                              16, 18, 20, 22, 24]
    points_img = bpf.create_grid_points_with_text(first_spot_grid_points,
                                                  spot_proportional_size=3,
                                                  image_size=(300, 200),
                                                  grid_size=(24, 24), 
                                                  spot_color=(0,0,0),
                                                  padding=30, 
                                                  background_color=(255,255,255),
                                                  text_color=(0, 0, 0), 
                                                  font_size=150,
                                                  show_text=True, 
                                                  num_columns=12,
                                                  txt_spacing=100,
                                                  min_padding_above_text=300)

    axs_points_img.imshow(points_img)
    axs_points_img.axis('off')
    axs_points_img.set_title("Distribution of points in the ROI\n(point number)")
    pat_pos = axs_points_img.get_position()
    new_pat_pos = [pat_pos.x0+xoffset, pat_pos.y0+yoffset, pat_pos.width*zoom,
                   pat_pos.height*zoom]
    axs_points_img.set_position(new_pat_pos)

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

def plot_image(image,axs_img,xoffset,yoffset,pltscale):
    axs_img.imshow(image, cmap='gray')
    pos = axs_img.get_position()  # Get the original position
    new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
               pos.height*pltscale]
    # Shrink the plot
    axs_img.set_position(new_pos)
    axs_img.axis('off')

def label_axis(axis_list, letter_label, xpos=-0.1, ypos=1, fontsize=16, fontweight='bold'):
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

def norm_values(cell_list,val_to_plot):
    cell_list = cell_list.copy()
    #print(f"cell list inside func : {cell_list}")
    cell_grp=cell_list.groupby(by="cell_ID")
    for c, cell in cell_grp:
        pat_grp = cell.groupby(by="frame_id")
        for p,pat in pat_grp:
            #print(f"c:{c}, p:{p}")
            pre_val= float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]=="pre")][val_to_plot])
            pp_grp = pat.groupby(by="pre_post_status")
            for pr, pp in pp_grp:
                norm_val = float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]==pr)][val_to_plot])
                norm_val = (norm_val/pre_val)*100
                cell_list.loc[(cell_list["cell_ID"]==c)&(cell_list["frame_id"]==p)&(cell_list["pre_post_status"]==pr),val_to_plot]=norm_val
    return cell_list
   
def del_values(cell_list,val_to_plot):
    cell_list = cell_list.copy()
    #print(f"cell list inside func : {cell_list}")
    cell_grp=cell_list.groupby(by="cell_ID")
    for c, cell in cell_grp:
        pat_grp = cell.groupby(by="frame_id")
        for p,pat in pat_grp:
            #print(f"c:{c}, p:{p}")
            pre_val= float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]=="pre")][val_to_plot])
            pp_grp = pat.groupby(by="pre_post_status")
            for pr, pp in pp_grp:
                del_val = float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]==pr)][val_to_plot])
                del_val = del_val-pre_val
                cell_list.loc[(cell_list["cell_ID"]==c)&(cell_list["frame_id"]==p)&(cell_list["pre_post_status"]==pr),val_to_plot]=del_val
    return cell_list

def norm_values_all_trials(cell_list, val_to_plot):
    cell_list = cell_list.copy()
    # Group by 'cell_ID'
    cell_grp = cell_list.groupby(by="cell_ID")
    for c, cell in cell_grp:
        pat_grp = cell.groupby(by="frame_id")
        for p, pat in pat_grp:
            trial_grp = pat.groupby(by="trial_no")
            for trial, trial_data in trial_grp:
                pre_val = trial_data[trial_data["pre_post_status"] == "pre"][val_to_plot].values
                if len(pre_val) == 0:
                    continue
                else:
                    pre_val = pre_val[0]  # Assuming there is only one 'pre' value per group
                    pp_grp = trial_data.groupby(by="pre_post_status")
                    for pr, pp in pp_grp:
                        norm_val = pp[val_to_plot].values
                        norm_val = (norm_val / pre_val) * 100
                        # Using vectorized operations for efficiency
                        mask_cell_ID = cell_list["cell_ID"] == c
                        mask_frame_id = cell_list["frame_id"] == p
                        mask_pre_post_status = cell_list["pre_post_status"] == pr
                        mask_trial_no = cell_list["trial_no"] == trial
                        combined_mask = mask_cell_ID & mask_frame_id & mask_pre_post_status & mask_trial_no
                        cell_list.loc[combined_mask, val_to_plot] = norm_val
    return cell_list



def plot_cell_dist(catcell_dist, val_to_plot, fig, axs, pattern_number, y_lim,
                   x_label, cell_type, plt_color, resp_color):
    pat_num = int(pattern_number.split("_")[-1])
    num_cells = len(catcell_dist["cell_ID"].unique())
    pfd = catcell_dist.groupby(by="frame_id")

    # Set a fixed y-axis limit for consistency across all plots
    axs.set_ylim(y_lim)

    for c, pat in pfd:
        if c != pattern_number:
            continue
        else:
            order = np.array(('pre', 'post_0', 'post_1', 'post_2', 'post_3'), dtype=object)
            
            # Scatter plot for cell response
            sns.stripplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}",
                order=order, ax=axs, color=resp_color,
                alpha=0.6, size=8, label='cell response'
            )

            # Point plot for average response
            sns.pointplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}", 
                errorbar="se", order=order, capsize=0.08, ax=axs,
                color=plt_color, linestyles='dotted', scale=0.8,
                label="average cell response"
            )

            # Calculate p-values for annotations
            pval_list = []
            pairs_list = []
            for i in order[1:]:
                result = spst.wilcoxon(
                    pat[pat["pre_post_status"] == 'pre'][f"{val_to_plot}"],
                    pat[pat["pre_post_status"] == i][f"{val_to_plot}"],
                    zero_method="wilcox", correction=True
                )
                pval_list.append(result.pvalue)
                pairs_list.append(("pre", i))

            # Manually add annotations using matplotlib
            if cell_type=="dep_cells":
                base_y = 400  # Absolute y-axis position for the first annotation
                step_y = 50    # Spacing between each annotation
            else:
                base_y = 400  # Absolute y-axis position for the first annotation
                step_y = 40    # Spacing between each annotation                             
            for idx, (pval, pair) in enumerate(zip(pval_list, pairs_list)):
                x1, x2 = pair
                x1_pos = order.tolist().index(x1)
                x2_pos = order.tolist().index(x2)
                
                # Draw the annotation line
                axs.plot([x1_pos, x2_pos], [base_y + idx * step_y] * 2, color='black', linewidth=1)
                
                # Add the p-value text above the line
                annotation_text = bpf.convert_pvalue_to_asterisks(pval)
                axs.text(
                    (x1_pos + x2_pos) / 2, base_y + idx * step_y + 2, 
                    annotation_text, ha='center', va='bottom', fontsize=10
                )

            # Draw a horizontal reference line at 100
            axs.axhline(100, ls=':', color="k", alpha=0.4)

            # Adjust axis labels and ticks
            if pat_num == 0:
                sns.despine(ax=axs, top=True, right=True)
                axs.set_ylabel("% change in\nPSH amplitude")
                axs.set_xlabel(None)
            elif pat_num == 1:
                sns.despine(ax=axs, top=True, right=True)
                axs.set_ylabel(None)
                axs.set_yticklabels([])
                if cell_type == "dep_cells":
                    axs.set_xlabel(x_label)
                else:
                    axs.set_xlabel(None)
            elif pat_num == 2:
                sns.despine(ax=axs, top=True, right=True)
                axs.set_xlabel(None)
                axs.set_ylabel(None)
                axs.set_yticklabels([])
            
            axs.set_xticklabels(time_points, rotation=0)

            # Remove legend if present
            if axs.get_legend() is not None:
                axs.get_legend().remove()

            if cell_type != "dep_cells":
                axs.set_xticklabels([])

    return axs





















#def plot_cell_dist(catcell_dist,val_to_plot,fig,axs,pattern_number,y_lim,
#                   x_label, cell_type,plt_color,resp_color):
#    pat_num=int(pattern_number.split("_")[-1])
#    num_cells= len(catcell_dist["cell_ID"].unique())
#    pfd = catcell_dist.groupby(by="frame_id")
#    for c, pat in pfd:
#        if c != pattern_number:
#            continue
#        else:
#            order = np.array(('pre','post_0','post_1','post_2','post_3'),dtype=object)
#            g=sns.stripplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
#                            order=order,ax=axs,color=resp_color,
#                            alpha=0.6,size=8, label='cell response')#alpha=0.8,
#            sns.pointplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
#                          errorbar="se",order=order,capsize=0.08,ax=axs,
#                          color=plt_color, linestyles='dotted',scale = 0.8,
#                         label="average cell response")
#            g.set_title(None)
#            #"""
#            pvalList = []
#            anotp_list = []
#            for i in order[1:]:
#                posti ="post{i}"
#                #non parametric, paired and small sample size, hence used Wilcoxon signed-rank test
#                #Wilcoxon signed-rank test
#                posti= spst.wilcoxon(pat[pat["pre_post_status"]=='pre'][f"{val_to_plot}"],pat[pat["pre_post_status"]==i][f"{val_to_plot}"],
#                                     zero_method="wilcox", correction=True)
#                pvalList.append(posti.pvalue)
#                anotp_list.append(("pre",i))
#            annotator = Annotator(axs,anotp_list,data=pat, 
#                                  x="pre_post_status",
#                                  y=f"{val_to_plot}",
#                                  order=order,
#                                 fontsize=10)
#            #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
#            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
#            annotator.annotate()
#            #"""
#            axs.axhline(100, ls=':',color="k", alpha=0.4)
#            if pat_num==0:
#                sns.despine(fig=None, ax=axs, top=True, right=True, 
#                            left=False, bottom=False, offset=None, trim=False)
#                axs.set_ylabel("% change in PSH")
#                axs.set_xlabel(None)
#                #axs[pat_num].set_yticks([])
#            elif pat_num==1:
#                sns.despine(fig=None, ax=axs, top=True, right=True, 
#                            left=False, bottom=False, offset=None, trim=False)
#                axs.set_ylabel(None)
#                axs.set_yticklabels([])
#            elif pat_num==2:
#                sns.despine(fig=None, ax=axs, top=True, right=True, 
#                            left=False, bottom=False, offset=None, trim=False)
#                axs.set_xlabel(None)
#                axs.set_ylabel(None)
#                axs.set_yticklabels([])
#            else:
#                pass 
#            g.set(ylim=y_lim)
#            g.set_xticklabels(time_points,rotation=0)
#            if g.get_legend() is not None:
#                    g.get_legend().remove()
#            if cell_type!="dep_cells":
#                axs.set_xticklabels([])
#            else:
#                pass 
#
#
#    ax_pos = axs.get_position()
#    #new_ax_pos = [ax_pos.x0-0.02, ax_pos.y0, ax_pos.width,
#    #              ax_pos.height]
#    #axs.set_position(new_ax_pos)

def plot_cell_category_classified_EPSP_features(esp_feat_cells_df,val_to_plot,
                                                fig,axs1,axs2,axs3,cell_type):
    cell_df= norm_values(esp_feat_cells_df,val_to_plot)
    if cell_type=="pot_cells":
        strp_color = bpf.CB_color_cycle[0]
        line_color = bpf.CB_color_cycle[5]
        y_lim = (-25,600)
        x_label = None
    elif cell_type=="dep_cells":
        strp_color = bpf.CB_color_cycle[1]
        line_color = bpf.CB_color_cycle[5]
        y_lim = (-25,1000)
        x_label = "time points (mins)"
    else:
        print("uncagerised cell")
    plot_cell_dist(cell_df,val_to_plot,fig,axs1,"pattern_0",
                   y_lim,x_label,cell_type,line_color,strp_color
                  )
    plot_cell_dist(cell_df,val_to_plot,fig,axs2,"pattern_1",
                   y_lim,x_label,cell_type,line_color,strp_color
                  )
    plot_cell_dist(cell_df,val_to_plot,fig,axs3,"pattern_2",
                   y_lim,x_label,cell_type,line_color,strp_color
                  )
    
def plot_response_summary_bar(sc_data_dict, fig, axs):
    order = ["pre", "post_3"]
    
    def preprocess_data(df, group):
        df = df.copy()
        df = norm_values(df, "min_trace")
        df = df[df["pre_post_status"] == "post_3"]
        df['group'] = group
        df['combined'] = group + '_' + df['frame_id'] + "_" + df['pre_post_status']
        return df[df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
    
    learners_df = preprocess_data(sc_data_dict["ap_cells"], 'learners')
    non_learners_df = preprocess_data(sc_data_dict["an_cells"], 'non_learners')
    combined_df = pd.concat([learners_df, non_learners_df])

    palette = {"learners": bpf.CB_color_cycle[0], "non_learners": bpf.CB_color_cycle[1]}

    sns.barplot(data=combined_df, x='frame_id', y='min_trace', hue='group', palette=palette, ax=axs, ci=None)

    grouped = combined_df.groupby(['combined', 'group'])['min_trace'].agg(['mean', 'sem']).reset_index()
    
    baseline = 100
    pval_list = [
        ttest_1samp(combined_df[combined_df['combined'] == group]['min_trace'], popmean=baseline)[1] 
        if len(combined_df[combined_df['combined'] == group]) > 0 else None 
        for group in grouped['combined'].unique()
    ]

    # Add error bars and p-values
    annotation_texts = [bpf.convert_pvalue_to_asterisks(p) for p in pval_list]
    for patch, row, annotation in zip(axs.patches, grouped.itertuples(), annotation_texts):
        bar_x = patch.get_x() + patch.get_width() / 2
        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
        
        if annotation:
            bar_y = patch.get_height() + 0.18 * patch.get_height()
            #bar_y = patch.get_height() + 10
            axs.text(bar_x, bar_y, annotation, ha='center', va='bottom', fontsize=10, color='black')

    # Formatting plot
    axs.axhline(baseline, linestyle=":", color="k", alpha=0.6)
    axs.set_ylabel("% change in\n PSH amplitude")
    axs.set_xlabel(None)
    #axs.set_xlabel("pattern type")
    
    # Change x tick labels
    axs.set_xticklabels(['trained', 'overlapping', 'non\noverlapping'],
                       rotation=15)
    
    axs.xaxis.set_ticks_position('none')
    axs.xaxis.set_tick_params(top=False)
    axs.yaxis.set_tick_params(right=False)
    axs.spines[['right', 'top']].set_visible(False)

    # Adjusting the legend
    if axs.legend_ is not None:
        legend = axs.legend(bbox_to_anchor=(0.6, 1.2), loc='center', frameon=False)
        legend.set_title('')
        legend.get_frame().set_alpha(0)  # Make legend background transparent

def plot_point_plasticity_dist(cell_features_all_trials, sc_data_dict, fig,
                               axs_lr, axs_nl):
    pre_color = bpf.pre_color
    lrn_post_color = bpf.CB_color_cycle[0]
    non_lrn_post_color = bpf.CB_color_cycle[1]
    
    # Normalize and filter data
    cell_features_all_trials["min_trace"] = cell_features_all_trials["min_trace"].apply(lambda x: np.nan if x > 5 else x)
    cell_features_all_trials = norm_values_all_trials(cell_features_all_trials, "min_trace")
    
    # Sort frame IDs and prepare x-tick labels
    order = cell_features_all_trials[cell_features_all_trials["frame_status"] == "point"]["frame_id"].unique()
    order = human_sort(order)
    x_ticklabels = [int(s.split("_")[-1]) + 1 if s[-1].isdigit() else None for s in order]

    learners_df = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners_df = sc_data_dict["an_cells"]["cell_ID"].unique()
    
    # Split data into learners and non-learners
    points_df_learners = cell_features_all_trials[cell_features_all_trials["cell_ID"].isin(learners_df)].copy()
    points_df_non_learners = cell_features_all_trials[cell_features_all_trials["cell_ID"].isin(non_learners_df)].copy()
    
    # Filter for 'pre' and 'post_3' statuses for learners and non-learners
    points_df_pre_learners = points_df_learners[(points_df_learners["frame_id"].str.contains("point")) & 
                                                (points_df_learners["pre_post_status"] == "pre")].reset_index(drop=True)
    points_df_post_learners = points_df_learners[(points_df_learners["frame_id"].str.contains("point")) & 
                                                 (points_df_learners["pre_post_status"] == "post_3")].reset_index(drop=True)

    points_df_pre_non_learners = points_df_non_learners[(points_df_non_learners["frame_id"].str.contains("point")) & 
                                                        (points_df_non_learners["pre_post_status"] == "pre")].reset_index(drop=True)
    points_df_post_non_learners = points_df_non_learners[(points_df_non_learners["frame_id"].str.contains("point")) & 
                                                         (points_df_non_learners["pre_post_status"] == "post_3")].reset_index(drop=True)
    
    # Plot learners
    sns.pointplot(data=points_df_pre_learners, x="frame_id", y="min_trace",
                  ax=axs_lr, color=pre_color, capsize=0.15,
                  order=order, errorbar='se')
    sns.pointplot(data=points_df_post_learners, x="frame_id", y="min_trace",
                  ax=axs_lr, color=lrn_post_color, capsize=0.15,
                  order=order, errorbar='se')
    sns.stripplot(data=points_df_pre_learners, x="frame_id", y="min_trace", 
                  ax=axs_lr, color=pre_color, alpha=0.2, order=order)
    sns.stripplot(data=points_df_post_learners, x="frame_id", y="min_trace", 
                  ax=axs_lr, color=lrn_post_color, alpha=0.2, order=order)

    # Plot non-learners
    sns.pointplot(data=points_df_pre_non_learners, x="frame_id", y="min_trace",
                  ax=axs_nl, color=pre_color, capsize=0.15,
                  order=order, errorbar='se')
    sns.pointplot(data=points_df_post_non_learners, x="frame_id", y="min_trace",
                  ax=axs_nl, color=non_lrn_post_color, capsize=0.15,
                  order=order, errorbar='se')
    sns.stripplot(data=points_df_pre_non_learners, x="frame_id", y="min_trace", 
                  ax=axs_nl, color=pre_color, alpha=0.2, order=order)
    sns.stripplot(data=points_df_post_non_learners, x="frame_id", y="min_trace", 
                  ax=axs_nl, color=non_lrn_post_color, alpha=0.2, order=order)
    
    # Define a fixed y-axis value for all annotations
    annotation_y_position = 400  # Adjust this value to set where annotations appear on the y-axis
    
    # Calculate and plot significance between 'pre' and 'post_3' for each frame_id
    for ax, pre_df, post_df in zip([axs_lr, axs_nl],
                                   [points_df_pre_learners, points_df_pre_non_learners],
                                   [points_df_post_learners, points_df_post_non_learners]):
        for frame_id in order:
            pre_values = pre_df[pre_df["frame_id"] == frame_id]["min_trace"]
            post_values = post_df[post_df["frame_id"] == frame_id]["min_trace"]

            # Perform Wilcoxon signed-rank test
            if len(pre_values) > 0 and len(post_values) > 0:
                p_value = spst.wilcoxon(pre_values, post_values, zero_method="wilcox", correction=True).pvalue

                # Convert p-value to asterisks
                annotation_text = bpf.convert_pvalue_to_asterisks(p_value)

                ## Annotate the plot at a fixed y-axis position
                #x_pos = order.index(frame_id)
                #ax.text(x_pos, annotation_y_position, annotation_text, 
                #        ha='center', va='bottom', fontsize=8, color='black')
    
    # Customize axes for learners
    axs_lr.set_ylim(-50, 500)
    axs_lr.set_ylabel("% change in\nPSH amplitude")
    axs_lr.set_xlabel("point no.")
    #axs_lr.set_xlabel(None)
    axs_lr.spines[['right', 'top']].set_visible(False)
    axs_lr.set_xticklabels(x_ticklabels)
    #axs_lr.set_xticklabels([])#x_ticklabels)
    #    #axs_lr.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, ncol=4)

    # Customize axes for non-learners
    axs_nl.set_ylim(-50, 500)
    axs_nl.set_xlabel("point no.")
    axs_nl.set_ylabel("% change in\n PSH amplitude")
    axs_nl.spines[['right', 'top']].set_visible(False)
    axs_nl.set_xticklabels(x_ticklabels)
    #axs_nl.set_ylabel(None)
    #axs_nl.set_yticklabels([])
    #axs_nl.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, ncol=4)




def plot_peak_perc_comp(
    sc_data_dict,
    axs_learners_pat_trained,
    axs_learners_pat_overlapping,
    axs_learners_pat_non_overlapping,
    axs_non_learners_pat_trained,
    axs_non_learners_pat_overlapping,
    axs_non_learners_pat_non_overlapping,
):
    order = ["pre", "post_3"]
    pattern_info = {
        "pattern_0": {"marker": "^", "label": "trained"},
        "pattern_1": {"marker": ".", "label": "overlapping"},
        "pattern_2": {"marker": "+", "label": "non\noverlapping"},
    }

    # Map patterns to specific axes for learners
    axis_map_learners = {
        "pattern_0": axs_learners_pat_trained,
        "pattern_1": axs_learners_pat_overlapping,
        "pattern_2": axs_learners_pat_non_overlapping,
    }

    # Map patterns to specific axes for non-learners
    axis_map_non_learners = {
        "pattern_0": axs_non_learners_pat_trained,
        "pattern_1": axs_non_learners_pat_overlapping,
        "pattern_2": axs_non_learners_pat_non_overlapping,
    }

    # Get data for learners
    learners_df = sc_data_dict["ap_cells"]
    learners_df = learners_df[learners_df["pre_post_status"].isin(order)]
    pat_df_learners = learners_df[
        learners_df["frame_id"].isin(pattern_info.keys())
    ].assign(group="learners")

    # Normalize data for learners
    norm_df_learners = del_values(pat_df_learners, "min_trace")

    # Plot for learners
    for pat in pattern_info.keys():
        pat_data = pat_df_learners[pat_df_learners["frame_id"] == pat]
        ax = axis_map_learners[pat]

        # Extract x and y data
        x = pat_data[pat_data["pre_post_status"] == "pre"]["min_trace"]
        y = norm_df_learners[
            (norm_df_learners["frame_id"] == pat)
            & (norm_df_learners["pre_post_status"] == "post_3")
        ]["min_trace"]

        if x.empty or y.empty:
            print(f"No data for learners - {pat}.")
            continue

        # Scatter plot
        color = bpf.CB_color_cycle[0]  # Color for learners
        ax.scatter(
            x,
            y,
            color=color,
            alpha=0.9,
            marker=pattern_info[pat]["marker"],
            label=pattern_info[pat]["label"],
        )

        # Add regression line
        if len(x) > 1:
            slope, intercept, _, _, _ = spst.linregress(x, y)
            x_line = np.linspace(-5, 10, 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color,
                alpha=0.7,
                linestyle="--",
            )

            # Compute Spearman correlation
            rho, p_value = spst.spearmanr(x, y)

            ## Annotate the plot with Spearman correlation coefficient and p-value
            #ax.annotate(
            #    f"ρ = {rho:.2f}\np = {p_value:.3f}",
            #    xy=(0.05, 0.9),
            #    xycoords='axes fraction',
            #    fontsize=10,
            #    verticalalignment='top',
            #    bbox=dict(facecolor='none', edgecolor='none')
            #    #bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5),
            #)

        # Set x and y limits
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.75, 2.5)

        # Adjust spines to hide right and top
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Set axis labels
        if pat == "pattern_0":
            ax.set_ylabel("change in\PSH amplitude(mV)")
        else:
            ax.set_ylabel("")

        if pat == "pattern_1":
            ax.set_xlabel("PSH amplitude\npre (mV)")
        else:
            ax.set_xlabel("")

        # Show y-tick labels only for pattern_0
        if pat != "pattern_0":
            ax.set_yticklabels([])

        # Ensure tick labels are visible
        ax.tick_params(axis="both", which="both", labelsize=10)

        # Set title based on pattern label
        ax.set_title(pattern_info[pat]['label'].capitalize(), fontsize=10)

        # Hide legends
        ax.legend().set_visible(False)

    # Get data for non-learners
    non_learners_df = sc_data_dict["an_cells"]
    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(order)]
    pat_df_non_learners = non_learners_df[
        non_learners_df["frame_id"].isin(pattern_info.keys())
    ].assign(group="non_learners")

    # Normalize data for non-learners
    norm_df_non_learners = del_values(pat_df_non_learners, "min_trace")

    # Plot for non-learners
    for pat in pattern_info.keys():
        pat_data = pat_df_non_learners[pat_df_non_learners["frame_id"] == pat]
        ax = axis_map_non_learners[pat]

        # Extract x and y data
        x = pat_data[pat_data["pre_post_status"] == "pre"]["min_trace"]
        y = norm_df_non_learners[
            (norm_df_non_learners["frame_id"] == pat)
            & (norm_df_non_learners["pre_post_status"] == "post_3")
        ]["min_trace"]

        if x.empty or y.empty:
            print(f"No data for non-learners - {pat}.")
            continue

        # Scatter plot
        color = bpf.CB_color_cycle[1]  # Color for non-learners
        ax.scatter(
            x,
            y,
            color=color,
            alpha=0.9,
            marker=pattern_info[pat]["marker"],
            label=pattern_info[pat]["label"],
        )

        # Add regression line
        if len(x) > 1:
            slope, intercept, _, _, _ = spst.linregress(x, y)
            x_line = np.linspace(-5, 10, 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color,
                alpha=0.7,
                linestyle="--",
            )

            # Compute Spearman correlation
            rho, p_value = spst.spearmanr(x, y)

            ## Annotate the plot with Spearman correlation coefficient and p-value
            #ax.annotate(
            #    f"ρ = {rho:.2f}\np = {p_value:.3f}",
            #    xy=(0.05, 0.9),
            #    xycoords='axes fraction',
            #    fontsize=10,
            #    verticalalignment='top',
            #    bbox=dict(facecolor='none', edgecolor='none')
            #    #bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5),
            #)

        # Set x and y limits
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.75, 2.5)

        # Adjust spines to hide right and top
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Set axis labels
        if pat == "pattern_0":
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")

        if pat == "pattern_1":
            ax.set_xlabel("PSH amplitude\npre (mV)")
        else:
            ax.set_xlabel("")

        # Hide y-tick labels
        ax.set_yticklabels([])

        # Ensure tick labels are visible
        ax.tick_params(axis="both", which="both", labelsize=10)

        # Set title based on pattern label
        ax.set_title(pattern_info[pat]['label'].capitalize(), fontsize=10)

        # Hide legends
        ax.legend().set_visible(False)




def plot_figure_5(extracted_feature_pickle_file_path,
                  PSH_illustration_path,
                  all_trial_path,
                  cell_categorised_pickle_file,
                  cell_stats_pickle_file,
                  outdir,learner_cell=learner_cell,
                  non_learner_cell=non_learner_cell):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    cell_stats_df = pd.read_hdf(cell_stats_pickle_file)
    cell_features_all_trials = pd.read_pickle(all_trial_path)
    print(f"cell stat df : {cell_stats_df}")
    single_cell_df = feature_extracted_data.copy()
    learner_cell_df = single_cell_df.copy()
    non_learner_cell_df = single_cell_df.copy()
    learner_cell_df = single_cell_df[(single_cell_df["cell_ID"]==learner_cell)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    non_learner_cell_df = single_cell_df[(single_cell_df["cell_ID"]==non_learner_cell)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    sc_data_dict = pd.read_pickle(cell_categorised_pickle_file)
    sc_data_df = pd.concat([sc_data_dict["ap_cells"],
                            sc_data_dict["an_cells"]]).reset_index(drop=True)
    print(f"sc data : {sc_data_df['cell_ID'].unique()}")
    
    psh_illust= pillow.Image.open(PSH_illustration_path)
    # Check if the image has an alpha channel (transparency)
    if psh_illust.mode == 'RGBA':
        # Create a white background using the correct Image class method
        white_bg = pillow.Image.new("RGB", psh_illust.size, (255, 255, 255))
        # Paste the image on top of the white background, handling transparency
        white_bg.paste(psh_illust, mask=psh_illust.split()[3])  # 3 is the alpha channel
        psh_illust = white_bg

    # If the image is not RGB, convert it to RGB
    if psh_illust.mode != "RGB":
        psh_illust = psh_illust.convert("RGB")

    
    # Define the width and height ratios
    height_ratios = [1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1
                     ]  # Adjust these values as needed
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1, 1, 1, 
                    1, 1]# Adjust these values as needed

    fig = plt.figure(figsize=(14,16))
    gs = GridSpec(22, 12,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.4, hspace=0.5)


    #plot illustration of PSH
    axs_illu = fig.add_subplot(gs[0:2,1:3])
    plot_image(psh_illust,axs_illu,0,0,1.5)
    move_axis([axs_illu],0.1,0.035,1)
    axs_illu.text(0.05,1.1,'A',transform=axs_illu.transAxes,
                 fontsize=16, fontweight='bold', ha='center', va='center')
    
    #plot patterns
    axs_pat_1 = fig.add_subplot(gs[0:1,1:2])
    axs_pat_2 = fig.add_subplot(gs[0:1,4:5])
    axs_pat_3 = fig.add_subplot(gs[0:1,7:8])
    plot_patterns(axs_pat_1,axs_pat_2,axs_pat_3,0,-0.04,1)

    #plot distribution epsp for learners and non-leaners
    axs_ex_pat1 = fig.add_subplot(gs[2:7,0:3])
    axs_ex_pat2 = fig.add_subplot(gs[2:7,3:6])
    axs_ex_pat3 = fig.add_subplot(gs[2:7,6:9])
    plot_cell_category_classified_EPSP_features(sc_data_dict["ap_cells"],
                                                "min_trace",fig,axs_ex_pat1,
                                                axs_ex_pat2,axs_ex_pat3,
                                                "pot_cells"
                                               )
    axs_ex_list = [axs_ex_pat1,axs_ex_pat2,axs_ex_pat3]
    label_axis(axs_ex_list,"B",xpos=-0.1, ypos=1.1)


    axs_in_pat1 = fig.add_subplot(gs[7:12,0:3])
    axs_in_pat2 = fig.add_subplot(gs[7:12,3:6])
    axs_in_pat3 = fig.add_subplot(gs[7:12,6:9])
    plot_cell_category_classified_EPSP_features(sc_data_dict["an_cells"],
                                                "min_trace",fig,axs_in_pat1,
                                                axs_in_pat2,axs_in_pat3,
                                                "dep_cells"
                                               )
    axs_in_list = [axs_in_pat1,axs_in_pat2,axs_in_pat3]
    label_axis(axs_in_list,"C",xpos=0.1, ypos=0.9)

    axs_bar = fig.add_subplot(gs[12:14,0:2])
    plot_response_summary_bar(sc_data_dict,fig,axs_bar)
    move_axis([axs_bar],0,-0.03,1)
    axs_bar.text(-0.1,1,'D',transform=axs_bar.transAxes,    
                 fontsize=16, fontweight='bold', ha='center',
                 va='center')
#    #comment here
#    axs_comp_per = fig.add_subplot(gs[18:20,0:3])
#    plot_peak_perc_comp(sc_data_dict,fig,axs_comp_per) 
#    move_axis([axs_comp_per],-0.0545,-0.075,1.75)
#    axs_comp_per.text(-0.05,1.05,'E',transform=axs_comp_per.transAxes,    
#                        fontsize=16, fontweight='bold', ha='center',
#                        va='center')
#    #till here
#    axs_comp_per_1 = fig.add_subplot(gs[16:18,0:1])
#    axs_comp_per_2 = fig.add_subplot(gs[16:18,2:3])
#    #plot_peak_perc_comp(sc_data_dict,fig,axs_comp_per) 
#    plot_peak_perc_comp(sc_data_dict, fig, axs_comp_per_1, axs_comp_per_2)
#    move_axis([axs_comp_per_1,axs_comp_per_2],-0.0545,-0.075,1.75)
#    label_axis([axs_comp_per_1,axs_comp_per_2],"E", xpos=-0.2, ypos=1.1)
#    #axs_comp_per.text(-0.05,1.05,'D',transform=axs_comp_per.transAxes,    
#    #                    fontsize=16, fontweight='bold', ha='center',
#    #                    va='center')
#    
#    axs_points_img = fig.add_subplot(gs[12:13,4:9])
#    plot_points(axs_points_img,-0.075,-0.08,zoom=2)
#
#    axs_points_lr = fig.add_subplot(gs[13:16,4:9])
#    axs_points_nl = fig.add_subplot(gs[16:19,4:9])
#    plot_point_plasticity_dist(cell_features_all_trials,sc_data_dict,fig,
#                               axs_points_lr,axs_points_nl)
#    move_axis([axs_points_lr,axs_points_nl],0,-0.075,1)
#    label_axis([axs_points_lr,axs_points_nl],"F", xpos=-0.05, ypos=1.05)    
#    #til here
    axs_learners_pat_trained = fig.add_subplot(gs[12:14,3:4])
    axs_learners_pat_overlapping = fig.add_subplot(gs[12:14,4:5]) 
    axs_learners_pat_non_overlappin = fig.add_subplot(gs[12:14,5:6])
    axs_non_learners_pat_trained = fig.add_subplot(gs[12:14,6:7])
    axs_non_learners_pat_overlapping  = fig.add_subplot(gs[12:14,7:8])
    axs_non_learners_pat_non_overlapping = fig.add_subplot(gs[12:14,8:9])
    plot_peak_perc_comp(sc_data_dict, 
                        axs_learners_pat_trained,
                        axs_learners_pat_overlapping, 
                        axs_learners_pat_non_overlappin,
                        axs_non_learners_pat_trained,
                        axs_non_learners_pat_overlapping, 
                        axs_non_learners_pat_non_overlapping)
    axs_scatr_list= [axs_learners_pat_trained,
                    axs_learners_pat_overlapping,
                    axs_learners_pat_non_overlappin,
                    axs_non_learners_pat_trained,
                    axs_non_learners_pat_overlapping,
                    axs_non_learners_pat_non_overlapping
                    ]
    move_axis(axs_scatr_list,0,-0.04,1)
    label_axis(axs_scatr_list,"E", xpos=-0.2, ypos=1.25)





#    #plot_peak_perc_comp(sc_data_dict,fig,axs_comp_per) 
#    plot_peak_perc_comp(sc_data_dict, fig, axs_comp_per_1, axs_comp_per_2)
#    move_axis([axs_comp_per_1,axs_comp_per_2],-0.0545,-0.075,1.75)
#    label_axis([axs_comp_per_1,axs_comp_per_2],"D", xpos=-0.2, ypos=1.1)
#    #axs_comp_per.text(-0.05,1.05,'D',transform=axs_comp_per.transAxes,    
#    #                    fontsize=16, fontweight='bold', ha='center',
#    #                    va='center')
#    
    axs_points_img_1 = fig.add_subplot(gs[15:16,0:4])
    plot_points(axs_points_img_1,-0.05,-0.09,zoom=1.7)
    axs_points_img_2 = fig.add_subplot(gs[15:16,5:9])
    plot_points(axs_points_img_2,-0.05,-0.09,zoom=1.7)#
    

    axs_points_lr = fig.add_subplot(gs[16:19,0:4])
    axs_points_nl = fig.add_subplot(gs[16:19,5:9])
    plot_point_plasticity_dist(cell_features_all_trials,sc_data_dict,fig,
                               axs_points_lr,axs_points_nl)
    move_axis([axs_points_lr,axs_points_nl],0,-0.075,1)
    label_axis([axs_points_lr,axs_points_nl],"F", xpos=-0.05, ypos=1.05)
    
    
    
#    #comment below
#    axs_points_img = fig.add_subplot(gs[14:15,4:9])
#    plot_points(axs_points_img,-0.075,-0.08,zoom=2)
#
#    axs_points_lr = fig.add_subplot(gs[15:18,4:9])
#    axs_points_nl = fig.add_subplot(gs[18:21,4:9])
#    plot_point_plasticity_dist(cell_features_all_trials,sc_data_dict,fig,
#                               axs_points_lr,axs_points_nl)
#    move_axis([axs_points_lr,axs_points_nl],0,-0.075,1)
#    label_axis([axs_points_lr,axs_points_nl],"F")
#    #till here
    
    #label_axis([axs_points_lr,axs_points_nl],"F")
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_5.png"
    #outpath = f"{outdir}/figure_5.svg"
    #outpath = f"{outdir}/figure_5.pdf"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser.
    description = '''Generates figure 5'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pikl-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with extracted features'
                       )
    parser.add_argument('--alltrial-path', '-t'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with extracted features'
                       )



    parser.add_argument('--sortedcell-path', '-s'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with cell sorted'
                        'exrracted data'
                       )
    parser.add_argument('--cellstat-path', '-c'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with cell sorted'
                        'exrracted data'
                       )
    parser.add_argument('--illustration-path', '-i'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file in png format'
                       )

    parser.add_argument('--outdir-path','-o'
                        ,required = False, default ='./', type=str
                        ,help = 'where to save the generated figure image'
                       )
    #    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    pklpath = Path(args.pikl_path)
    scpath = Path(args.sortedcell_path)
    PSH_illustration_path = Path(args.illustration_path)
    cell_stat_path = Path(args.cellstat_path)
    all_trial_df_path = Path(args.alltrial_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_5'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_5(pklpath,PSH_illustration_path,
                  all_trial_df_path,scpath,cell_stat_path,globoutdir)
    #print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
