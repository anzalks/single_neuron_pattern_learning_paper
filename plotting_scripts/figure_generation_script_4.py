__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 4 of pattern learning paper.
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
import baisic_plot_fuctnions_and_features as bpf
import re
from scipy.stats import ttest_1samp
from scipy.stats import spearmanr

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

def label_axis(axis_list, letter_label, xpos=-0.1, ypos=1.1, fontsize=16, fontweight='bold'):
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
                base_y = 200  # Absolute y-axis position for the first annotation
                step_y = 20    # Spacing between each annotation
            else:
                base_y = 450  # Absolute y-axis position for the first annotation
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
                    annotation_text, ha='center', va='bottom', fontsize=8
                )

            # Draw a horizontal reference line at 100
            axs.axhline(100, ls=':', color="k", alpha=0.4)

            # Adjust axis labels and ticks
            if pat_num == 0:
                sns.despine(ax=axs, top=True, right=True)
                axs.set_ylabel("% change in\nEPSP amplitude")
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
#                                 fontsize=8)
#            #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
#            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
#            annotator.annotate()
#            #"""
#            axs.axhline(100, ls=':',color="k", alpha=0.4)
#            if pat_num==0:
#                sns.despine(fig=None, ax=axs, top=True, right=True, 
#                            left=False, bottom=False, offset=None, trim=False)
#                axs.set_ylabel("% change in\nEPSP amplitude")
#                axs.set_xlabel(None)
#                #axs[pat_num].set_yticks([])
#            elif pat_num==1:
#                sns.despine(fig=None, ax=axs, top=True, right=True, 
#                            left=False, bottom=False, offset=None, trim=False)
#                axs.set_ylabel(None)
#                axs.set_yticklabels([])
#                if cell_type=="dep_cells":
#                    axs.set_xlabel(x_label)
#                else:
#                    axs.set_xlabel(None)
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
        y_lim = (0,700)
        x_label = None
    elif cell_type=="dep_cells":
        strp_color = bpf.CB_color_cycle[1]
        line_color = bpf.CB_color_cycle[5]
        y_lim = (-5,300)
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



#def plot_response_summary_bar(sc_data_dict, fig, axs):
#    order = ["pre", "post_3"]
#    
#    # Normalize values for learners
#    learners_df = sc_data_dict["ap_cells"]
#    learners_df = norm_values(learners_df, "max_trace")
#    learners_df = learners_df[learners_df["pre_post_status"].isin(["post_3"])]
#    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#
#    # Normalize values for non-learners
#    non_learners_df = sc_data_dict["an_cells"]
#    non_learners_df = norm_values(non_learners_df, "max_trace")
#    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(["post_3"])]
#    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#
#    # Add a column to distinguish between learners and non-learners
#    pat_df_learners['group'] = 'learners'
#    pat_df_non_learners['group'] = 'non_learners'
#
#    # Create the combined column for x-axis
#    pat_df_learners['combined'] = 'learners_' + pat_df_learners['frame_id'] + "_" + pat_df_learners['pre_post_status']
#    pat_df_non_learners['combined'] = 'non_learners_' + pat_df_non_learners['frame_id'] + "_" + pat_df_non_learners['pre_post_status']
#
#    # Define the order of the combined x-axis categories
#    combined_order = ['learners_pattern_0_post_3', 
#                      'learners_pattern_1_post_3',
#                      'learners_pattern_2_post_3',
#                      'non_learners_pattern_0_post_3', 
#                      'non_learners_pattern_1_post_3',
#                      'non_learners_pattern_2_post_3']
#
#    x_labels = ['trained', 'overlapping', 'non\noverlapping',
#                'trained', 'overlapping', 'non\noverlapping']
#
#    # Concatenate the two DataFrames
#    combined_df = pd.concat([pat_df_learners, pat_df_non_learners])
#
#    # Define the color palette
#    palette = {"learners": bpf.CB_color_cycle[0], "non_learners": bpf.CB_color_cycle[1]}
#
#    # Plot 'post_3' bars
#    bars = sns.barplot(data=combined_df[combined_df["pre_post_status"] == "post_3"],
#                       x="combined", y="max_trace", hue="group", order=combined_order,
#                       palette=palette, alpha=1, ax=axs, ci=None)
#
#    # Move non-learners (orange) bars slightly to the left
#    for i, patch in enumerate(bars.patches):
#        # Shift non-learners bars (orange ones) by checking the patch color
#        if i >= len(combined_order) // 2:  # Assumes non-learners bars are the second half
#            patch.set_x(patch.get_x() - 0.15)  # Move the bar slightly to the left
#
#    # Align error bars by grouping patches based on hue
#    grouped = combined_df.groupby(['combined', 'group'])['max_trace'].agg(['mean', 'sem']).reset_index()
#    for i, (patch, row) in enumerate(zip(bars.patches, grouped.itertuples())):
#        bar_x = patch.get_x() + patch.get_width() / 2
#        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
#
#    # Customize x-axis
#    axs.set_xticklabels(x_labels, rotation=90, ha="center")
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.axhline(100, linestyle=":", color="k", alpha=0.6)  # Dashed baseline
#
#    # Customize labels
#    axs.set_ylabel("% change in\nEPSP amplitude")
#    axs.set_xlabel(None)
#    axs.xaxis.set_ticks_position('none')
#    axs.legend_.remove()




def plot_response_summary_bar(sc_data_dict, fig, axs):
    order = ["pre", "post_3"]
    
    def preprocess_data(df, group):
        df = df.copy()
        df = norm_values(df, "max_trace")
        df = df[df["pre_post_status"] == "post_3"]
        df['group'] = group
        df['combined'] = group + '_' + df['frame_id'] + "_" + df['pre_post_status']
        return df[df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
    
    learners_df = preprocess_data(sc_data_dict["ap_cells"], 'learners')
    non_learners_df = preprocess_data(sc_data_dict["an_cells"], 'non_learners')
    combined_df = pd.concat([learners_df, non_learners_df])

    palette = {"learners": bpf.CB_color_cycle[0], "non_learners": bpf.CB_color_cycle[1]}

    sns.barplot(data=combined_df, x='frame_id', y='max_trace', hue='group', palette=palette, ax=axs, ci=None)

    grouped = combined_df.groupby(['combined', 'group'])['max_trace'].agg(['mean', 'sem']).reset_index()
    
    baseline = 100
    pval_list = [
        ttest_1samp(combined_df[combined_df['combined'] == group]['max_trace'], popmean=baseline)[1] 
        if len(combined_df[combined_df['combined'] == group]) > 0 else None 
        for group in grouped['combined'].unique()
    ]

    # Add error bars and p-values
    annotation_texts = [bpf.convert_pvalue_to_asterisks(p) for p in pval_list]
    for patch, row, annotation in zip(axs.patches, grouped.itertuples(), annotation_texts):
        bar_x = patch.get_x() + patch.get_width() / 2
        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
        
        if annotation:
            bar_y = patch.get_height() + 10
            axs.text(bar_x, bar_y, annotation, ha='center', va='bottom', fontsize=10, color='black')

    # Formatting plot
    axs.axhline(baseline, linestyle=":", color="k", alpha=0.6)
    axs.set_ylabel("% change in\nEPSP amplitude")
    axs.xaxis.set_ticks_position('none')
    axs.xaxis.set_tick_params(top=False)
    axs.yaxis.set_tick_params(right=False)
    axs.spines[['right', 'top']].set_visible(False)

    # Adjusting the legend
    if axs.legend_ is not None:
        legend = axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        legend.set_title('')
        legend.get_frame().set_alpha(0)  # Make legend background transparent


def plot_response_summary_bar(sc_data_dict, fig, axs):
    order = ["pre", "post_3"]
    
    def preprocess_data(df, group):
        df = df.copy()
        df = norm_values(df, "max_trace")
        df = df[df["pre_post_status"] == "post_3"]
        df['group'] = group
        df['combined'] = group + '_' + df['frame_id'] + "_" + df['pre_post_status']
        return df[df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
    
    learners_df = preprocess_data(sc_data_dict["ap_cells"], 'learners')
    non_learners_df = preprocess_data(sc_data_dict["an_cells"], 'non_learners')
    combined_df = pd.concat([learners_df, non_learners_df])

    palette = {"learners": bpf.CB_color_cycle[0], "non_learners": bpf.CB_color_cycle[1]}

    sns.barplot(data=combined_df, x='frame_id', y='max_trace', hue='group', palette=palette, ax=axs, ci=None)

    grouped = combined_df.groupby(['combined', 'group'])['max_trace'].agg(['mean', 'sem']).reset_index()
    
    baseline = 100
    pval_list = [
        ttest_1samp(combined_df[combined_df['combined'] == group]['max_trace'], popmean=baseline)[1] 
        if len(combined_df[combined_df['combined'] == group]) > 0 else None 
        for group in grouped['combined'].unique()
    ]

    # Add error bars and p-values
    annotation_texts = [bpf.convert_pvalue_to_asterisks(p) for p in pval_list]
    for patch, row, annotation in zip(axs.patches, grouped.itertuples(), annotation_texts):
        bar_x = patch.get_x() + patch.get_width() / 2
        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
        
        if annotation:
            bar_y = patch.get_height() + 10
            axs.text(bar_x, bar_y, annotation, ha='center', va='bottom', fontsize=10, color='black')

    # Formatting plot
    axs.axhline(baseline, linestyle=":", color="k", alpha=0.6)
    axs.set_ylabel("% change in\nEPSP amplitude")
    axs.set_xlabel(None)
    #axs.set_xlabel("pattern type")
    
    # Change x tick labels
    axs.set_xticklabels(['trained', 'overlapping', 'non\noverlapping'])
    
    axs.xaxis.set_ticks_position('none')
    axs.xaxis.set_tick_params(top=False)
    axs.yaxis.set_tick_params(right=False)
    axs.spines[['right', 'top']].set_visible(False)

    # Adjusting the legend
    if axs.legend_ is not None:
        legend = axs.legend(bbox_to_anchor=(1.05, 0.9), loc='center', frameon=False)
        legend.set_title('')
        legend.get_frame().set_alpha(0)  # Make legend background transparent

##everythign fixed but learners and nonleraners are on same x axis labels.
#def plot_response_summary_bar(sc_data_dict, fig, axs):
#    order = ["pre", "post_3"]
#    
#    # Function to preprocess data (normalizing and filtering)
#    def preprocess_data(df, group):
#        df = df.copy()
#        df = norm_values(df, "max_trace")
#        df = df[df["pre_post_status"] == "post_3"]
#        df['group'] = group
#        df['combined'] = group + '_' + df['frame_id'] + "_" + df['pre_post_status']
#        return df[df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#    
#    # Apply preprocessing to both learners and non-learners
#    learners_df = preprocess_data(sc_data_dict["ap_cells"], 'learners')
#    non_learners_df = preprocess_data(sc_data_dict["an_cells"], 'non_learners')
#
#    # Combine DataFrames
#    combined_df = pd.concat([learners_df, non_learners_df])
#
#    # Define the order for combined categories
#    combined_order = ['learners_pattern_0_post_3', 'learners_pattern_1_post_3', 'learners_pattern_2_post_3',
#                      'non_learners_pattern_0_post_3', 'non_learners_pattern_1_post_3', 'non_learners_pattern_2_post_3']
#
#    # Define color palette
#    palette = {"learners": bpf.CB_color_cycle[0], "non_learners": bpf.CB_color_cycle[1]}
#
#    # Calculate means and standard errors for error bars
#    grouped = combined_df.groupby(['combined', 'group'])['max_trace'].agg(['mean', 'sem']).reset_index()
#
#    baseline = 100
#    pval_list = [ttest_1samp(combined_df[combined_df['combined'] == group]['max_trace'], popmean=baseline)[1] 
#                 if len(combined_df[combined_df['combined'] == group]) > 0 else None for group in combined_order]
#
#    bar_width = 0.25
#
#    # Plot learners using sns.barplot
#    sns.barplot(data=combined_df[combined_df['group'] == 'learners'], x='combined', y='max_trace', 
#                color=palette["learners"], order=combined_order[:3], ax=axs, ci=None)
#
#    # Add error bars for learners
#    for i, row in enumerate(grouped.iloc[:3].itertuples()):
#        bar_x = axs.patches[i].get_x() + bar_width / 2
#        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
#
#    # Plot non-learners using sns.barplot
#    sns.barplot(data=combined_df[combined_df['group'] == 'non_learners'], x='combined', y='max_trace', 
#                color=palette["non_learners"], order=combined_order[3:], ax=axs, ci=None)
#
#    # Add error bars for non-learners
#    for i, row in enumerate(grouped.iloc[3:].itertuples()):
#        bar_x = axs.patches[i + 3].get_x() + bar_width / 2
#        axs.errorbar(bar_x, row.mean, yerr=row.sem, fmt='none', c='black', capsize=5)
#
#    # Add p-value annotations with adjusted vertical spacing
#    annotation_texts = [bpf.convert_pvalue_to_asterisks(p) for p in pval_list]
#    for i, annotation in enumerate(annotation_texts):
#        if annotation:
#            bar_x = axs.patches[i].get_x() + bar_width / 2
#            bar_y = axs.patches[i].get_height() + 10
#            axs.text(bar_x, bar_y, annotation, ha='center', va='bottom', fontsize=10, color='black')
#
#    # Formatting plot
#    axs.set_xticks(range(len(combined_order)))
#    axs.set_xticklabels(['trained', 'overlapping', 'non\noverlapping', 'trained', 'overlapping', 'non\noverlapping'], 
#                        rotation=90, ha="center")
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.axhline(baseline, linestyle=":", color="k", alpha=0.6)
#    axs.set_ylabel("% change in\nEPSP amplitude")
#    axs.xaxis.set_ticks_position('none')
#    axs.xaxis.set_tick_params(top=False)
#    axs.yaxis.set_tick_params(right=False)
#
#    if axs.legend_ is not None:
#        axs.legend_.remove()

#def plot_response_summary_bar(sc_data_dict,fig,axs):
#    order = ["pre", "post_3"]
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique
#    learners_df = sc_data_dict["ap_cells"]
#    learners_df= norm_values(learners_df,"max_trace")
#    learners_df = learners_df[learners_df["pre_post_status"].isin(["post_3"])]
#    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0",
#                                                                "pattern_1",
#                                                                "pattern_2"])]
#
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique
#    non_learners_df = sc_data_dict["an_cells"]
#    non_learners_df= norm_values(non_learners_df,"max_trace")
#    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(["post_3"])]
#    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1",
#                                                      "pattern_2"])]
#
#    # Add a column to distinguish between learners and non-learners
#    pat_df_learners['group'] = 'learners'
#    pat_df_non_learners['group'] = 'non_learners'
#
#    # Create the combined column for x-axis
#    pat_df_learners['combined'] = 'learners_' + pat_df_learners['frame_id'] +"_" + pat_df_learners['pre_post_status']
#    pat_df_non_learners['combined'] = 'non_learners_' + pat_df_non_learners['frame_id'] + "_" +pat_df_non_learners['pre_post_status']
#
#    # Define the order of the combined x-axis categories
#    combined_order = ['learners_pattern_0_post_3', 
#                      'learners_pattern_1_post_3',
#                      'learners_pattern_2_post_3',
#                      'non_learners_pattern_0_post_3', 
#                      'non_learners_pattern_1_post_3',
#                      'non_learners_pattern_2_post_3']
#
##    x_labels = [None, 'trained\npattern', None,'overlapping\npattern',None,
##               'non-overlapping\npattern',
##                None, 'trained\npattern', None,'overlapping\npattern',None,
##                'non-overlapping\npattern'
##               ]
#    x_labels = ['trained', 'overlapping',
#               'non\noverlapping',
#                'trained', 'overlapping',
#                'non\noverlapping',
#               ]
#
#
#    #x_labels = ['trained\npattern', None,'overlapping\npattern',None,
#    #           'non-overlapping\npattern',
#    #            None, 'trained\npattern', None,'overlapping\npattern',None,
#    #            'non-overlapping\npattern', None
#    #           ]
#    # Concatenate the two DataFrames
#    combined_df = pd.concat([pat_df_learners, pat_df_non_learners])
#
#    # Define the color palette
#    palette = {"learners": bpf.CB_color_cycle[0], 
#               "non_learners":bpf.CB_color_cycle[1]}
#
#    # Plotting with seaborn
#
#    # Plot 'pre' bars
#    #sns.barplot(data=combined_df[combined_df["pre_post_status"] == "pre"],
#    #            x="combined", y="max_trace", hue="group", order=combined_order,
#    #            palette=palette, alpha=0.5,ax=axs,errorbar=None)
#
#    # Plot 'post_3' bars
#    sns.barplot(data=combined_df[combined_df["pre_post_status"] == "post_3"],
#                x="combined", y="max_trace", hue="group", order=combined_order,
#                palette=palette,alpha=1,ax=axs,ci=None)
#    #for label in axs.get_xticklabels():
#    #    label.set_position((label.get_position()[0] + 0.5, label.get_position()[1]))  # Adjust the offset value as needed
#
#    axs.set_xticklabels(x_labels)
#    axs.set_xticklabels(axs.get_xticklabels(),rotation=90, ha="right")
#    axs.spines[['right', 'top']].set_visible(False)
#    #axs.tick_params(axis='x', pad=5)
#    axs.axhline(100,linestyle=":",color="k",alpha=0.6)
#    axs.legend_.remove()
#    axs.set_ylabel("% change in\nEPSP amplitude")
#    axs.set_xlabel(None)
#    axs.xaxis.set_ticks_position('none')
#    #axs.set_ylim(-2,10,)



#significance added to point plots and fixed the annotator distance
def plot_point_plasticity_dist(cell_features_all_trials, sc_data_dict, fig,
                               axs_lr, axs_nl):
    pre_color = bpf.pre_color
    lrn_post_color = bpf.CB_color_cycle[0]
    non_lrn_post_color = bpf.CB_color_cycle[1]
    
    # Normalize and filter data
    cell_features_all_trials["max_trace"] = cell_features_all_trials["max_trace"].apply(lambda x: np.nan if x > 5 else x)
    cell_features_all_trials = norm_values_all_trials(cell_features_all_trials, "max_trace")
    
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
    sns.pointplot(data=points_df_pre_learners, x="frame_id", y="max_trace",
                  ax=axs_lr, color=pre_color, label='Learners - Pre', capsize=0.15,
                  order=order, errorbar='se')
    sns.pointplot(data=points_df_post_learners, x="frame_id", y="max_trace",
                  ax=axs_lr, color=lrn_post_color, label='Learners - Post', capsize=0.15,
                  order=order, errorbar='se')
    sns.stripplot(data=points_df_pre_learners, x="frame_id", y="max_trace", 
                  ax=axs_lr, color=pre_color, alpha=0.2, order=order)
    sns.stripplot(data=points_df_post_learners, x="frame_id", y="max_trace", 
                  ax=axs_lr, color=lrn_post_color, alpha=0.2, order=order)

    # Plot non-learners
    sns.pointplot(data=points_df_pre_non_learners, x="frame_id", y="max_trace",
                  ax=axs_nl, color=pre_color, label='Non-Learners - Pre', capsize=0.15,
                  order=order, errorbar='se')
    sns.pointplot(data=points_df_post_non_learners, x="frame_id", y="max_trace",
                  ax=axs_nl, color=non_lrn_post_color, label='Non-Learners - Post', capsize=0.15,
                  order=order, errorbar='se')
    sns.stripplot(data=points_df_pre_non_learners, x="frame_id", y="max_trace", 
                  ax=axs_nl, color=pre_color, alpha=0.2, order=order)
    sns.stripplot(data=points_df_post_non_learners, x="frame_id", y="max_trace", 
                  ax=axs_nl, color=non_lrn_post_color, alpha=0.2, order=order)
    
    # Define a fixed y-axis value for all annotations
    annotation_y_position = 400  # Adjust this value to set where annotations appear on the y-axis
    
    # Calculate and plot significance between 'pre' and 'post_3' for each frame_id
    for ax, pre_df, post_df in zip([axs_lr, axs_nl],
                                   [points_df_pre_learners, points_df_pre_non_learners],
                                   [points_df_post_learners, points_df_post_non_learners]):
        for frame_id in order:
            pre_values = pre_df[pre_df["frame_id"] == frame_id]["max_trace"]
            post_values = post_df[post_df["frame_id"] == frame_id]["max_trace"]

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
    axs_lr.set_ylabel("% change in\nEPSP amplitude")
    axs_lr.set_xlabel(None)
    axs_lr.spines[['right', 'top']].set_visible(False)
    axs_lr.set_xticklabels([])#x_ticklabels)
    axs_lr.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, ncol=4)

    # Customize axes for non-learners
    axs_nl.set_ylim(-50, 500)
    axs_nl.set_xlabel("point no.")
    axs_nl.set_ylabel("% change in\nEPSP amplitude")
    axs_nl.spines[['right', 'top']].set_visible(False)
    axs_nl.set_xticklabels(x_ticklabels)
    #axs_nl.set_ylabel(None)
    #axs_nl.set_yticklabels([])
    axs_nl.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, ncol=4)

#def plot_point_plasticity_dist(cell_features_all_trials, sc_data_dict, fig,
#                               axs_lr,axs_nl):
#    pre_color= bpf.pre_color 
#    lrn_post_color = bpf.CB_color_cycle[0]
#    non_lrn_post_color = bpf.CB_color_cycle[1]
#    cell_features_all_trials["max_trace"] = cell_features_all_trials["max_trace"].apply(lambda x: np.nan if x > 5 else x)
#    cell_features_all_trials=norm_values_all_trials(cell_features_all_trials,
#                                                    "max_trace")
#    order = cell_features_all_trials[cell_features_all_trials["frame_status"]=="point"]["frame_id"].unique()
#    order = human_sort(order)
#    x_ticklabels = [int(s.split("_")[-1])+1 if s[-1].isdigit() else None for s in order]
#
#    learners_df = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners_df = sc_data_dict["an_cells"]["cell_ID"].unique()
#    
#    # Split data into learners and non-learners
#    points_df_learners = cell_features_all_trials[cell_features_all_trials["cell_ID"].isin(learners_df)].copy()
#    points_df_non_learners = cell_features_all_trials[cell_features_all_trials["cell_ID"].isin(non_learners_df)].copy()
#    
#    # Filter for 'pre' and 'post_3' statuses
#    points_df_pre_learners = points_df_learners[(points_df_learners["frame_id"].str.contains("point")) &(points_df_learners["pre_post_status"] == "pre")].reset_index(drop=True)
#    points_df_post_learners = points_df_learners[(points_df_learners["frame_id"].str.contains("point"))&(points_df_learners["pre_post_status"] =="post_3")].reset_index(drop=True)
#
#    points_df_pre_non_learners = points_df_non_learners[(points_df_non_learners["frame_id"].str.contains("point")) & (points_df_non_learners["pre_post_status"] == "pre")].reset_index(drop=True)
#    points_df_post_non_learners = points_df_non_learners[(points_df_non_learners["frame_id"].str.contains("point")) & (points_df_non_learners["pre_post_status"] == "post_3")].reset_index(drop=True)
#    
#    # Plot learners
#    sns.pointplot(data=points_df_pre_learners, x="frame_id", y="max_trace",
#                  ax=axs_lr, color=pre_color, label='Learners - Pre',capsize=0.15,
#                  order=order, errorbar='se')
#    sns.pointplot(data=points_df_post_learners, x="frame_id", y="max_trace",
#                  ax=axs_lr, color=lrn_post_color, label='Learners - Post',capsize=0.15,
#                  order=order , errorbar='se')
#    sns.stripplot(data=points_df_pre_learners, x="frame_id", y="max_trace", 
#                  ax=axs_lr, color=pre_color, alpha=0.2,order=order)
#    sns.stripplot(data=points_df_post_learners, x="frame_id", y="max_trace", 
#                  ax=axs_lr, color=lrn_post_color, alpha=0.2,order=order)
#    # Plot non-learners
#    sns.pointplot(data=points_df_pre_non_learners, x="frame_id", y="max_trace",
#                  ax=axs_nl, color=pre_color, label='Non-Learners - Pre',
#                  capsize=0.15, order=order, errorbar='se')
#    sns.pointplot(data=points_df_post_non_learners, x="frame_id",
#                  y="max_trace", ax=axs_nl, color=non_lrn_post_color,order=order, 
#                  label='Non-Learners - Post',capsize=0.15, errorbar='se')
#    sns.stripplot(data=points_df_pre_non_learners, x="frame_id", y="max_trace", 
#                  ax=axs_nl, color=pre_color, alpha=0.2,order=order)
#    sns.stripplot(data=points_df_post_non_learners, x="frame_id", y="max_trace", 
#                  ax=axs_nl, color=non_lrn_post_color, alpha=0.2, order=order)
#    # Customization
#    #axs_lr.set_ylim(-0.1, 4)
#    axs_lr.set_ylim(-50,500)
#    axs_lr.set_ylabel("% change in\nEPSP amplitude")
#    axs_lr.set_xlabel("point no.")
#    axs_lr.spines[['right', 'top']].set_visible(False)
#    axs_lr.set_xticklabels(x_ticklabels)
#    axs_lr.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False,
#                  ncol=4)
#    #axs_nl.set_ylim(-0.1, 4)
#    axs_nl.set_ylim(-50,500)
#    axs_nl.set_xlabel("point no.")
#    axs_lr.set_xlabel("point no.")
#    axs_nl.spines[['right', 'top']].set_visible(False)
#    axs_nl.set_xticklabels(x_ticklabels)
#    axs_nl.set_ylabel(None)
#    axs_nl.set_yticklabels([])
#    axs_nl.legend(loc='upper center', bbox_to_anchor=(0.5, 1),frameon=False, 
#                  ncol=4)


def plot_peak_comp_pre_post(sc_data_dict,fig,axs):
    order = ["pre", "post_3"]
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique
    learners_df = sc_data_dict["ap_cells"]
    #learners_df= norm_values(learners_df,"max_trace")
    learners_df = learners_df[learners_df["pre_post_status"].isin(["pre",
                                                                   "post_3"])]
    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0",
                                                                "pattern_1",
                                                                "pattern_2"])]
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique
    non_learners_df = sc_data_dict["an_cells"]
    #non_learners_df= norm_values(non_learners_df,"max_trace")
    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(["pre", "post_3"])]
    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1",
                                                      "pattern_2"])]

    # Add a column to distinguish between learners and non-learners
    pat_df_learners['group'] = 'learners'
    pat_df_non_learners['group'] = 'non_learners'
    all_cell_df = pd.concat([pat_df_learners,pat_df_non_learners])
    lrn_grp = all_cell_df.groupby(by="group")
    for lrn,lrn_data in lrn_grp:
        if lrn=="learners":
            color=bpf.CB_color_cycle[0]
            labl="lr"
        else:
            color=bpf.CB_color_cycle[1]
            labl="n_lr"
        pat_grp = lrn_data.groupby(by="frame_id")
        for pat,pat_data in pat_grp:
            if pat=="pattern_0":
                alpha=0.5
                marker="^"
                label = f"trained"#_{labl}"
            elif pat=="pattern_1":
                alpha=0.5
                marker="."
                label = f"overlapping"#_{labl}"
            elif pat=="pattern_2":
                alpha=0.5
                marker="+"
                label = f"non-overlapping"#_{labl}"
            x= pat_data[pat_data["pre_post_status"]=="pre"]["max_trace"]
            y= pat_data[pat_data["pre_post_status"]=="post_3"]["max_trace"]
            axs.scatter(x,y,color=color,alpha=alpha,marker=marker,label=label)
    axs.axline([0, 0], [1, 1],alpha=0.5,color='k',linestyle=":")
    axs.set_aspect("equal")
    axs.set_ylim(-1,10)
    axs.set_xlim(-1,10)
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_ylabel("EPSP amplitude post\n30 min(mV)")
    axs.set_xlabel("EPSP amplitude pre (mV)")
    axs.legend(loc='upper center', bbox_to_anchor=(1.4, 1),frameon=False, 
                  ncol=1)

#def plot_peak_perc_comp(sc_data_dict,fig,axs):
#    order = ["pre", "post_3"]
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique
#    learners_df = sc_data_dict["ap_cells"]
#    #learners_df= norm_values(learners_df,"max_trace")
#    learners_df = learners_df[learners_df["pre_post_status"].isin(["pre",
#                                                                   "post_3"])]
#    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0",
#                                                                "pattern_1",
#                                                                "pattern_2"])]
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique
#    non_learners_df = sc_data_dict["an_cells"]
#    #non_learners_df= norm_values(non_learners_df,"max_trace")
#    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(["pre", "post_3"])]
#    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1",
#                                                      "pattern_2"])]
#
#    # Add a column to distinguish between learners and non-learners
#    pat_df_learners['group'] = 'learners'
#    pat_df_non_learners['group'] = 'non_learners'
#    all_cell_df = pd.concat([pat_df_learners,pat_df_non_learners])
#    norm_df = norm_values(all_cell_df,"max_trace")
#    lrn_grp = all_cell_df.groupby(by="group")
#    for lrn,lrn_data in lrn_grp:
#        if lrn=="learners":
#            color=bpf.CB_color_cycle[0]
#            labl="lr"
#        else:
#            color=bpf.CB_color_cycle[1]
#            labl="n_lr"
#        pat_grp = lrn_data.groupby(by="frame_id")
#        for pat,pat_data in pat_grp:
#            if pat=="pattern_0":
#                alpha=0.5
#                marker="^"
#                label = f"trained"#_{labl}"
#            elif pat=="pattern_1":
#                alpha=0.5
#                marker="."
#                label = f"overlapping"#_{labl}"
#            elif pat=="pattern_2":
#                alpha=0.5
#                marker="+"
#                label = f"untrained"#_{labl}"
#            x= pat_data[pat_data["pre_post_status"]=="pre"]["max_trace"]
#            y=norm_df[(norm_df["group"]==lrn)&(norm_df["frame_id"]==pat)&(norm_df["pre_post_status"]=="post_3")]["max_trace"]
#            axs.scatter(x,y,color=color,alpha=alpha,marker=marker,label=label)
#    axs.axline([0, -50], [10, 500],alpha=0.5,color='k', linestyle=":")
#    axs.set_ylim(-50,500)
#    axs.set_xlim(0,10)
#    axs.set_aspect(0.02)
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.set_ylabel("% LTP post-training")
#    axs.set_xlabel("EPSP amplitude pre (mV)")
#    axs.legend(loc='upper center', bbox_to_anchor=(1.4, 0.9),frameon=False, 
#                  ncol=1)
#


#def plot_peak_perc_comp(sc_data_dict, fig, axs_learners, axs_non_learners):
#    order = ["pre", "post_3"]
#    
#    # Get data for learners
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique
#    learners_df = sc_data_dict["ap_cells"]
#    learners_df = learners_df[learners_df["pre_post_status"].isin(order)]
#    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#    
#    # Get data for non-learners
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique
#    non_learners_df = sc_data_dict["an_cells"]
#    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(order)]
#    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#
#    # Add a column to distinguish between learners and non-learners
#    pat_df_learners['group'] = 'learners'
#    pat_df_non_learners['group'] = 'non_learners'
#    
#    # Combine data and normalize
#    all_cell_df = pd.concat([pat_df_learners, pat_df_non_learners])
#    norm_df = norm_values(all_cell_df, "max_trace")
#    
#    # Group by 'group' (learners and non-learners)
#    lrn_grp = all_cell_df.groupby(by="group")
#    
#    # Plot learners and non-learners separately
#    for lrn, lrn_data in lrn_grp:
#        if lrn == "learners":
#            color = bpf.CB_color_cycle[0]
#            labl = "lr"
#            ax = axs_learners  # Use the axis for learners
#        else:
#            color = bpf.CB_color_cycle[1]
#            labl = "n_lr"
#            ax = axs_non_learners  # Use the axis for non-learners
#        
#        # Group by 'frame_id'
#        pat_grp = lrn_data.groupby(by="frame_id")
#        for pat, pat_data in pat_grp:
#            if pat == "pattern_0":
#                alpha = 0.5
#                marker = "^"
#                label = f"trained"
#            elif pat == "pattern_1":
#                alpha = 0.5
#                marker = "."
#                label = f"overlapping"
#            elif pat == "pattern_2":
#                alpha = 0.5
#                marker = "+"
#                label = f"untrained"
#            
#            # Extract pre and post-training data
#            x = pat_data[pat_data["pre_post_status"] == "pre"]["max_trace"]
#            y = norm_df[(norm_df["group"] == lrn) & 
#                        (norm_df["frame_id"] == pat) & 
#                        (norm_df["pre_post_status"] == "post_3")]["max_trace"]
#            
#            # Plot on the respective axis
#            ax.scatter(x, y, color=color, alpha=alpha, marker=marker, label=label)
#        
#        # Customize the axes
#        ax.axline([0, -50], [10, 500], alpha=0.5, color='k', linestyle=":")
#        ax.set_ylim(-50, 500)
#        ax.set_xlim(0, 10)
#        ax.set_aspect(0.02)
#        ax.spines[['right', 'top']].set_visible(False)
#        ax.set_ylabel("% LTP post-training")
#        ax.set_xlabel("EPSP amplitude pre (mV)")
#        ax.legend(loc='upper center', bbox_to_anchor=(1.4, 0.9), frameon=False, ncol=1)


#def plot_peak_perc_comp(sc_data_dict, fig, axs_learners, axs_non_learners):
#    order = ["pre", "post_3"]
#    
#    # Get data for learners
#    learners_df = sc_data_dict["ap_cells"]
#    learners_df = learners_df[learners_df["pre_post_status"].isin(order)]
#    pat_df_learners = learners_df[learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#    
#    # Get data for non-learners
#    non_learners_df = sc_data_dict["an_cells"]
#    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(order)]
#    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(["pattern_0", "pattern_1", "pattern_2"])]
#
#    # Add a column to distinguish between learners and non-learners
#    pat_df_learners['group'] = 'learners'
#    pat_df_non_learners['group'] = 'non_learners'
#    
#    # Combine data and normalize
#    all_cell_df = pd.concat([pat_df_learners, pat_df_non_learners])
#    norm_df = norm_values(all_cell_df, "max_trace")
#    
#    # Group by 'group' (learners and non-learners)
#    lrn_grp = all_cell_df.groupby(by="group")
#    
#    # Plot learners and non-learners separately
#    for lrn, lrn_data in lrn_grp:
#        if lrn == "learners":
#            color = bpf.CB_color_cycle[0]
#            ax = axs_learners
#        else:
#            color = bpf.CB_color_cycle[1]
#            ax = axs_non_learners
#        
#        # Group by 'frame_id'
#        pat_grp = lrn_data.groupby(by="frame_id")
#        for pat, pat_data in pat_grp:
#            if pat == "pattern_0":
#                alpha = 0.9
#                marker = "^"
#                label = "trained"
#            elif pat == "pattern_1":
#                alpha = 0.9
#                marker = "."
#                label = "overlapping"
#            elif pat == "pattern_2":
#                alpha = 0.9
#                marker = "+"
#                label = "untrained"
#            
#            # Extract pre and post-training data
#            x = pat_data[pat_data["pre_post_status"] == "pre"]["max_trace"]
#            y = norm_df[(norm_df["group"] == lrn) & 
#                        (norm_df["frame_id"] == pat) & 
#                        (norm_df["pre_post_status"] == "post_3")]["max_trace"]
#            
#            # Plot on the respective axis
#            ax.scatter(x, y, color=color, alpha=alpha, marker=marker, label=label)
#        
#        # Customize the axes
#        ax.axline([0, -50], [10, 500], alpha=0.5, color='k', linestyle=":")
#        ax.set_ylim(-50, 500)
#        ax.set_xlim(0, 10)
#        ax.set_aspect(0.02)
#        ax.spines[['right', 'top']].set_visible(False)
#        ax.set_xlabel("EPSP amplitude\npre (mV)")
#        
#        if lrn == "learners":
#            ax.set_ylabel("% LTP post-training")
#        else:
#            # Remove y-axis features for non-learners
#            ax.set_ylabel("")
#            ax.set_yticklabels([])
#        
#        # Adjust the legend position
#        ax.legend(
#            loc='center', 
#            bbox_to_anchor=(0.5, 1.3), 
#            frameon=True, 
#            ncol=1, 
#            handletextpad=0.05, 
#            framealpha=0.5  # Adjust frame transparency
#        )


def plot_peak_perc_comp(sc_data_dict, fig, axs_learners, axs_non_learners):
    order = ["pre", "post_3"]
    pattern_info = {
        "pattern_0": {"marker": "^", "label": "trained"},
        "pattern_1": {"marker": ".", "label": "overlapping"},
        "pattern_2": {"marker": "+", "label": "untrained"}
    }
    
    # Get data for learners and non-learners
    learners_df = sc_data_dict["ap_cells"]
    non_learners_df = sc_data_dict["an_cells"]
    
    learners_df = learners_df[learners_df["pre_post_status"].isin(order)]
    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(order)]
    
    pat_df_learners = learners_df[learners_df["frame_id"].isin(pattern_info.keys())].assign(group='learners')
    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(pattern_info.keys())].assign(group='non_learners')
    
    # Combine and normalize data
    all_cell_df = pd.concat([pat_df_learners, pat_df_non_learners], ignore_index=True)
    norm_df = del_values(all_cell_df, "max_trace")

    # Group data and plot
    for (lrn, pat), pat_data in all_cell_df.groupby(["group", "frame_id"]):
        color = bpf.CB_color_cycle[0] if lrn == "learners" else bpf.CB_color_cycle[1]
        ax = axs_learners if lrn == "learners" else axs_non_learners
        
        x = pat_data[pat_data["pre_post_status"] == "pre"]["max_trace"]
        y = norm_df[(norm_df["group"] == lrn) & 
                    (norm_df["frame_id"] == pat) & 
                    (norm_df["pre_post_status"] == "post_3")]["max_trace"]
        
        if x.empty or y.empty:
            print(f"No data for {lrn} - {pat}.")
            continue
        
        ax.scatter(x, y, color=color, alpha=0.9, 
                   marker=pattern_info[pat]["marker"], 
                   label=pattern_info[pat]["label"])
        
        ax.set_ylim(-3.5, 4)
        ax.set_xlim(0, 8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("EPSP amplitude\npre (mV)")
        
        if lrn == "learners":
            ax.set_ylabel("% LTP post-training")
        else:
            ax.set_yticklabels([])
        
        # Legend customization
        ax.legend(loc='center', bbox_to_anchor=(0.6, 1.2), frameon=True,
                  ncol=1)


def plot_peak_perc_comp(sc_data_dict, fig, axs_learners, axs_non_learners):
    order = ["pre", "post_3"]
    pattern_info = {
        "pattern_0": {"marker": "^", "label": "trained"},
        "pattern_1": {"marker": ".", "label": "overlapping"},
        "pattern_2": {"marker": "+", "label": "non-overlapping"}
    }
    
    # Get data for learners and non-learners
    learners_df = sc_data_dict["ap_cells"]
    non_learners_df = sc_data_dict["an_cells"]
    
    learners_df = learners_df[learners_df["pre_post_status"].isin(order)]
    non_learners_df = non_learners_df[non_learners_df["pre_post_status"].isin(order)]
    
    pat_df_learners = learners_df[learners_df["frame_id"].isin(pattern_info.keys())].assign(group='learners')
    pat_df_non_learners = non_learners_df[non_learners_df["frame_id"].isin(pattern_info.keys())].assign(group='non_learners')
    
    # Combine and normalize data
    all_cell_df = pd.concat([pat_df_learners, pat_df_non_learners], ignore_index=True)
    norm_df = del_values(all_cell_df, "max_trace")

    # Group data and plot
    for (lrn, pat), pat_data in all_cell_df.groupby(["group", "frame_id"]):
        color = bpf.CB_color_cycle[0] if lrn == "learners" else bpf.CB_color_cycle[1]
        ax = axs_learners if lrn == "learners" else axs_non_learners
        
        # Extract x and y data
        x = pat_data[pat_data["pre_post_status"] == "pre"]["max_trace"]
        y = norm_df[(norm_df["group"] == lrn) & 
                    (norm_df["frame_id"] == pat) & 
                    (norm_df["pre_post_status"] == "post_3")]["max_trace"]
        
        if x.empty or y.empty:
            print(f"No data for {lrn} - {pat}.")
            continue
        
        # Perform Spearman correlation test
        corr_coeff, p_value = spearmanr(x, y)
        
        # Scatter plot
        ax.scatter(x, y, color=color, alpha=0.9, 
                   marker=pattern_info[pat]["marker"], 
                   label=pattern_info[pat]["label"])
        
        if pat=="pattern_0":
            # Display correlation coefficient and p-value on the plot
            ax.text(0.05, -0.4, f"trained\nSpearman r={corr_coeff:.2f} p={p_value:.3f}", 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        if pat=="pattern_1":
            # Display correlation coefficient and p-value on the plot                       
            ax.text(0.05, -0.6, f"overlapping\nSpearman r={corr_coeff:.2f} p={p_value:.3f}", 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        if pat=="pattern_2":
            # Display correlation coefficient and p-value on the plot
            ax.text(0.05, -0.8, f"non-overlapping\nSpearman r={corr_coeff:.2f} p={p_value:.3f}", 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


        # Customize axes
        ax.set_ylim(-3.5, 4)
        ax.set_xlim(0, 8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("EPSP amplitude\npre (mV)")
        
        if lrn == "learners":
            ax.set_ylabel("% LTP post-training")
        else:
            ax.set_yticklabels([])
        
        # Legend customization
        ax.legend(loc='center', bbox_to_anchor=(0.6, 1.1), 
                  frameon=True,ncol=1)

def plot_figure_4(extracted_feature_pickle_file_path,
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
    # Define the width and height ratios
    height_ratios = [1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1
                     ]  # Adjust these values as needed
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1, 1, 1, 
                    1, 1]# Adjust these values as needed

    fig = plt.figure(figsize=(14,16))
    gs = GridSpec(20, 12,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.4, hspace=0.5)



    #plot patterns
    axs_pat_1 = fig.add_subplot(gs[0:1,1:2])
    axs_pat_2 = fig.add_subplot(gs[0:1,4:5])
    axs_pat_3 = fig.add_subplot(gs[0:1,7:8])
    plot_patterns(axs_pat_1,axs_pat_2,axs_pat_3,0,-0.05,1)

    #plot distribution epsp for learners and non-leaners
    axs_ex_pat1 = fig.add_subplot(gs[2:7,0:3])
    axs_ex_pat2 = fig.add_subplot(gs[2:7,3:6])
    axs_ex_pat3 = fig.add_subplot(gs[2:7,6:9])
    plot_cell_category_classified_EPSP_features(sc_data_dict["ap_cells"],
                                                "max_trace",fig,axs_ex_pat1,
                                                axs_ex_pat2,axs_ex_pat3,
                                                "pot_cells"
                                               )
    axs_ex_list = [axs_ex_pat1,axs_ex_pat2,axs_ex_pat3]
    label_axis(axs_ex_list,"A", xpos=-0.1, ypos=1.1)


    axs_in_pat1 = fig.add_subplot(gs[7:12,0:3])
    axs_in_pat2 = fig.add_subplot(gs[7:12,3:6])
    axs_in_pat3 = fig.add_subplot(gs[7:12,6:9])
    plot_cell_category_classified_EPSP_features(sc_data_dict["an_cells"],
                                                "max_trace",fig,axs_in_pat1,
                                                axs_in_pat2,axs_in_pat3,
                                                "dep_cells"
                                               )
    axs_in_list = [axs_in_pat1,axs_in_pat2,axs_in_pat3]
    label_axis(axs_in_list,"B", xpos=0.1, ypos=0.9)

    axs_bar = fig.add_subplot(gs[12:14,0:3])
    plot_response_summary_bar(sc_data_dict,fig,axs_bar)
    move_axis([axs_bar],0,-0.05,1)
    axs_bar.text(-0.05,1.05,'C',transform=axs_bar.transAxes,    
                 fontsize=16, fontweight='bold', ha='center',
                 va='center')

    #axs_comp_peaks = fig.add_subplot(gs[12:14,4:6])
    #plot_peak_comp_pre_post(sc_data_dict,fig,axs_comp_peaks)
    #move_axis([axs_comp_peaks],-0.055,-0.1,1.75)
    #axs_comp_peaks.text(-0.05,1.05,'D',transform=axs_comp_peaks.transAxes,    
    #             fontsize=16, fontweight='bold', ha='center',
    #             va='center')
    axs_comp_per_1 = fig.add_subplot(gs[16:18,0:1])
    axs_comp_per_2 = fig.add_subplot(gs[16:18,2:3])
    #plot_peak_perc_comp(sc_data_dict,fig,axs_comp_per) 
    plot_peak_perc_comp(sc_data_dict, fig, axs_comp_per_1, axs_comp_per_2)
    move_axis([axs_comp_per_1,axs_comp_per_2],-0.0545,-0.075,1.75)
    label_axis([axs_comp_per_1,axs_comp_per_2],"D", xpos=-0.2, ypos=1.1)
    #axs_comp_per.text(-0.05,1.05,'D',transform=axs_comp_per.transAxes,    
    #                    fontsize=16, fontweight='bold', ha='center',
    #                    va='center')
    
    axs_points_img = fig.add_subplot(gs[12:13,4:9])
    plot_points(axs_points_img,-0.075,-0.08,zoom=2)

    axs_points_lr = fig.add_subplot(gs[13:16,4:9])
    axs_points_nl = fig.add_subplot(gs[16:19,4:9])
    plot_point_plasticity_dist(cell_features_all_trials,sc_data_dict,fig,
                               axs_points_lr,axs_points_nl)
    move_axis([axs_points_lr,axs_points_nl],0,-0.075,1)
    label_axis([axs_points_lr,axs_points_nl],"E", xpos=-0.05, ypos=1.05)














    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    

    plt.tight_layout()
    outpath = f"{outdir}/figure_4.png"
    #outpath = f"{outdir}/figure_4.svg"
    #outpath = f"{outdir}/figure_4.pdf"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser.
    description = '''Generates figure 4'''
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
    illustration_path = Path(args.illustration_path)
    cell_stat_path = Path(args.cellstat_path)
    all_trial_df_path = Path(args.alltrial_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_4'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_4(pklpath,all_trial_df_path,scpath,cell_stat_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
