__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 3 of pattern learning paper.
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
from matplotlib.ticker import MultipleLocator
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm
from scipy.stats import ks_2samp, mannwhitneyu
import pingouin as pg
from matplotlib.lines import Line2D

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
cell_dist_key = ["learners","non-learners","cells not\nconsidered"]

class Args: pass
args_ = Args()

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
            pat_fr = bpf.create_grid_image(0,2)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,2)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,2)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        pat_pos = axs_pat.get_position()
        new_pat_pos = [pat_pos.x0+xoffset, pat_pos.y0+yoffset, pat_pos.width,
                        pat_pos.height]
        axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        axs_pat.set_title(pattern,fontsize=10)



def plot_pie_cell_dis(fig, axs, cell_dist, cell_dist_key):
    palette_color = sns.color_palette('colorblind')
    
    # Create the pie chart and get references to text objects
    wedges, texts, autotexts = axs.pie(cell_dist, 
                                       labels=cell_dist_key,
                                       colors=palette_color,
                                       startangle=210, 
                                       labeldistance=1.15, 
                                       autopct='%.0f%%')

    # Change the color of the text inside the pie chart to white
    for autotext in autotexts:
        autotext.set_color('white')  # Set the color to white

    # Optionally, adjust the text size if needed
    for text in autotexts:
        text.set_fontsize(12)  # Example size adjustment

    return axs
    

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

#def plot_threshold_timing(training_data,sc_data_dict,fig,axs):
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
#    cell_grp = training_data.groupby(by="cell_ID")
#    lrns=[]
#    non_lrns=[]
#    for cell, cell_data in cell_grp:
#        if cell in learners:
#            lrns.append(cell_data)
#        elif cell in non_learners:
#            non_lrns.append(cell_data)
#        else:
#            print(cell,"no selection")
#            continue
#    #print(lrns)
#    #print(non_lrns)
#    lrns = pd.concat(lrns)
#    lrns["l_stat"]="learners"
#    non_lrns = pd.concat(non_lrns)
#    non_lrns["l_stat"]="non\nlearners"
#    all_df = pd.concat([lrns,non_lrns])
#    print(non_lrns["trigger_time"].unique())
#    palette = {"learners": bpf.CB_color_cycle[0], "non\nlearners": bpf.CB_color_cycle[1]}
#    sns.pointplot(data=all_df,x="l_stat",
#                  y="cell_thresh_time",
#                  hue="l_stat",
#                  palette=palette,ci="sd",
#                  capsize=0.15, dodge=True,ax=axs)
##    
##    sns.pointplot(data=all_df[all_df["l_stat"]=="non_learner"],x="l_stat",
##                  y="cell_thresh_time",
##                  color=bpf.CB_color_cycle[1],ci="sd" ,
##                  capsize=0.15,dodge=True,ax=axs)
##
#    sns.stripplot(data=all_df,
#                  x="l_stat", y="cell_thresh_time",
#                  hue="l_stat",
#                  palette=palette,alpha=0.5)
##    sns.stripplot(data=all_df[all_df["l_stat"]=="non_learner"] ,
##                  x="l_stat", y="cell_thresh_time",
##                  color=bpf.CB_color_cycle[1],alpha=0.5)
#    axs.set_ylabel("rise time from \nprojection (ms)")
#    axs.set_xlabel(None)
#    axs.set_ylim(0,20)
#    axs.spines[['right', 'top']].set_visible(False)
#    handles, labels = axs.get_legend_handles_labels()
#    by_label = dict(zip(["learners","non-learners"], handles))
#    fig.legend(by_label.values(), by_label.keys(), 
#               bbox_to_anchor =(0.5, 0.32),
#               ncol = 6,
#               loc='upper center')#,frameon=False)#,loc='lower center'
#    axs.legend_.remove()

#violin plot

def plot_threshold_timing(training_data, sc_data_dict, fig, axs):
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
    cell_grp = training_data.groupby(by="cell_ID")
    lrns = []
    non_lrns = []
    
    # Separate learners and non-learners
    for cell, cell_data in cell_grp:
        if cell in learners:
            lrns.append(cell_data)
        elif cell in non_learners:
            non_lrns.append(cell_data)
        else:
            print(cell, "no selection")
            continue

    # Concatenate learners and non-learners dataframes
    lrns = pd.concat(lrns)
    lrns["l_stat"] = "learners"
    non_lrns = pd.concat(non_lrns)
    non_lrns["l_stat"] = "non\nlearners"
    all_df = pd.concat([lrns, non_lrns])

    # Set palette for learners and non-learners
    palette = {"learners": bpf.CB_color_cycle[0], "non\nlearners": bpf.CB_color_cycle[1]}
    
    # Violin plot for learners and non-learners
    sns.violinplot(
        data=all_df, x="l_stat", y="cell_thresh_time", hue="l_stat",
        palette=palette, alpha=0.6, inner="quartile", linewidth=1, ax=axs
    )

    # Customize labels, limits, and hide spines
    axs.set_ylabel("rise time from \nprojection (ms)")
    axs.set_xlabel(None)
    axs.set_ylim(0, 20)
    axs.set_xlim(-0.5, 1.75)
    axs.spines[['right', 'top']].set_visible(False)

    # Perform Mann-Whitney U test to compare the distributions
    learners_data = all_df[all_df["l_stat"] == "learners"]["cell_thresh_time"]
    non_learners_data = all_df[all_df["l_stat"] == "non\nlearners"]["cell_thresh_time"]
    
    # Mann-Whitney U test
    stat_test = spst.mannwhitneyu(learners_data, non_learners_data, alternative='two-sided')
    pval = stat_test.pvalue

    # Convert p-value to asterisks
    pval_asterisks = bpf.convert_pvalue_to_asterisks(pval)

    # Annotate the p-value on the plot
    annot = Annotator(axs, [("learners", "non\nlearners")], data=all_df,
                      x="l_stat", y="cell_thresh_time", palette=palette)
    
    # Set custom annotations using the asterisks conversion and then annotate
    annot.set_custom_annotations([pval_asterisks])
    annot.annotate()

    # Get legend handles and labels
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Set a custom legend
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.32),
               ncol=6, loc='upper center')

    # Remove the axis legend
    axs.legend_.remove()



#stripplot alone
#def plot_threshold_timing(training_data, sc_data_dict, fig, axs):
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
#    cell_grp = training_data.groupby(by="cell_ID")
#    lrns = []
#    non_lrns = []
#    
#    # Separate learners and non-learners
#    for cell, cell_data in cell_grp:
#        if cell in learners:
#            lrns.append(cell_data)
#        elif cell in non_learners:
#            non_lrns.append(cell_data)
#        else:
#            print(cell, "no selection")
#            continue
#
#    # Concatenate learners and non-learners dataframes
#    lrns = pd.concat(lrns)
#    lrns["l_stat"] = "learners"
#    non_lrns = pd.concat(non_lrns)
#    non_lrns["l_stat"] = "non\nlearners"
#    all_df = pd.concat([lrns, non_lrns])
#
#    # Set palette for learners and non-learners
#    palette = {"learners": bpf.CB_color_cycle[0], "non\nlearners": bpf.CB_color_cycle[1]}
#    
#    # Pointplot for mean +/- SD
#    sns.pointplot(data=all_df, x="l_stat", y="cell_thresh_time", hue="l_stat",
#                  palette=palette, ci="sd", capsize=0.15, dodge=True, ax=axs)
#
#    # Stripplot for individual points
#    sns.stripplot(data=all_df, x="l_stat", y="cell_thresh_time",
#                  palette=palette, alpha=0.5, ax=axs, dodge=True)
#
#    # Customize labels, limits, and hide spines
#    axs.set_ylabel("rise time from \nprojection (ms)")
#    axs.set_xlabel(None)
#    axs.set_ylim(0, 20)
#    axs.set_xlim(-0.5,1.75)
#    axs.spines[['right', 'top']].set_visible(False)
#
#    # Perform Mann-Whitney U test to compare the distributions
#    learners_data = all_df[all_df["l_stat"] == "learners"]["cell_thresh_time"]
#    non_learners_data = all_df[all_df["l_stat"] == "non\nlearners"]["cell_thresh_time"]
#    
#    # Mann-Whitney U test
#    stat_test = spst.mannwhitneyu(learners_data, non_learners_data, alternative='two-sided')
#    pval = stat_test.pvalue
#
#    # Convert p-value to asterisks
#    pval_asterisks = bpf.convert_pvalue_to_asterisks(pval)
#
#    # Annotate the p-value on the plot
#    annot = Annotator(axs, [("learners", "non\nlearners")], data=all_df,
#                      x="l_stat", y="cell_thresh_time", palette=palette)
#    
#    # Set custom annotations using the asterisks conversion and then annotate
#    annot.set_custom_annotations([pval_asterisks])
#    annot.annotate()
#
#    # Get legend handles and labels
#    handles, labels = axs.get_legend_handles_labels()
#    by_label = dict(zip(labels, handles))
#
#    # Set a custom legend
#    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.32),
#               ncol=6, loc='upper center')
#
#    # Remove the axis legend
#    axs.legend_.remove()

















































#def plot_mini_feature(cells_df,field_to_plot, learners,non_learners,fig,axs):
#    if field_to_plot=="mepsp_amp":
#        ylim=(-1,2)
#        ylabel=("mEPSP amplitude (mV)")
#    elif field_to_plot=="freq_mepsp":
#        ylim=(-1,10)
#        ylabel=("mEPSP frequency (Hz)")
#    else:
#        ylim=(None,None)
#        ylabel=None
#    order = np.array(["pre","post_3"])
#    cells_df=cells_df.copy()
#    data_to_plot = cells_df[cells_df["pre_post_status"].isin(["pre","post_3"])]
#    learners_df =  data_to_plot[data_to_plot["cell_ID"].isin(learners)].reset_index(drop=True)
#    non_learners_df =  data_to_plot[data_to_plot["cell_ID"].isin(non_learners)].reset_index(drop=True)
#    pre_dat =  learners_df[learners_df["pre_post_status"]=="pre"][field_to_plot].reset_index(drop=True)
#    post_dat =  learners_df[learners_df["pre_post_status"]=="post_3"][field_to_plot].reset_index(drop=True)
#
#    pointplot1=sns.pointplot(data=learners_df,x="pre_post_status",y=field_to_plot,ax=axs,
#                 order=order,color=bpf.CB_color_cycle[0],
#                 capsize=0.15,ci='sd')
#    pointplot2=sns.pointplot(data=non_learners_df,x="pre_post_status",
#                  y=field_to_plot,ax=axs,order=order,
#                  color=bpf.CB_color_cycle[1],capsize=0.15,
#                  ci='sd')
#    #sns.stripplot(data=learners_df,x="pre_post_status",y=field_to_plot,ax=axs,
#    #             order=order,color=bpf.CB_color_cycle[0],
#    #             alpha=0.1,zorder=1)
#    #sns.stripplot(data=non_learners_df,x="pre_ost_status",
#    #              y=field_to_plot,ax=axs,order=order,
#    #              color=bpf.CB_color_cycle[1],alpha=0.1,zorder=1)
#
#
#    stat_analysis= spst.wilcoxon(pre_dat,post_dat, zero_method="wilcox", correction=True)
#    pvalList = stat_analysis.pvalue
#    anotp_list=["pre","post_3"]
#    annotator =Annotator(axs,[anotp_list], data=learners_df, 
#                         x="pre_post_status",y=field_to_plot,order=order)
#    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
#    annotator.annotate()
#    axs.set_ylabel(ylabel)
#    axs.set_xlabel("time points\n(mins)")
#    axs.set_xticklabels(["pre","30 mins"])
#    axs.set_ylim(ylim)
#    axs.set_xlim(-0.5, 1.75)
#    axs.spines[['right', 'top']].set_visible(False)
#    return None



#from matplotlib.lines import Line2D
#import scipy.stats as spst
#import seaborn as sns
#import numpy as np

def plot_mini_feature(cells_df, field_to_plot, learners, non_learners, fig, axs):
    # Set y-axis limits and labels based on the field to plot
    if field_to_plot == "mepsp_amp":
        ylim = (-1, 2)
        ylabel = "mEPSP amplitude (mV)"
    elif field_to_plot == "freq_mepsp":
        ylim = (-1, 10)
        ylabel = "mEPSP frequency (Hz)"
    else:
        ylim = (None, None)
        ylabel = None

    # Define the order of the time points
    order = np.array(["pre", "post_3"])
    cells_df = cells_df.copy()
    data_to_plot = cells_df[cells_df["pre_post_status"].isin(order)]

    # Function to filter outliers using the IQR rule
    def remove_outliers(df, field):
        q1 = df[field].quantile(0.25)
        q3 = df[field].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[field] >= lower_bound) & (df[field] <= upper_bound)]

    # Separate learners and non-learners and remove outliers
    learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(learners)], field_to_plot).reset_index(drop=True)
    non_learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(non_learners)], field_to_plot).reset_index(drop=True)

    # Combine learners and non-learners into a single DataFrame for split violin plot
    combined_df = pd.concat([learners_df, non_learners_df])
    combined_df['group'] = combined_df['cell_ID'].apply(lambda x: 'learner' if x in learners else 'non-learner')

    # Create split violin plot without Seaborn's automatic legend
    sns.violinplot(
        data=combined_df, 
        x="pre_post_status", 
        y=field_to_plot, 
        hue="group", 
        ax=axs,
        order=order, 
        split=True, 
        inner="quartile",
        palette={'learner': bpf.CB_color_cycle[0], 'non-learner': bpf.CB_color_cycle[1]}, 
        linewidth=1,
        legend=False  # Disable automatic legend
    )

    # Remove any existing legend
    if axs.legend_:
        axs.legend_.remove()

    # Annotate comparison within learners (pre vs. post_3)
    stat_analysis = spst.mannwhitneyu(
        learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot],
        learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot],
        alternative='two-sided'
    )
    pval_within_learners = stat_analysis.pvalue
    annotator_within = Annotator(axs, [("pre", "post_3")], data=learners_df,
                                 x="pre_post_status", y=field_to_plot, order=order)
    annotator_within.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pval_within_learners)])
    annotator_within.annotate()

    # Perform comparisons between learners and non-learners at each time point
    learners_pre = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot]
    non_learners_pre = non_learners_df[non_learners_df["pre_post_status"] == "pre"][field_to_plot]
    learners_post_3 = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot]
    non_learners_post_3 = non_learners_df[non_learners_df["pre_post_status"] == "post_3"][field_to_plot]

    # Perform Mann-Whitney U tests for both time points
    stat_test_pre = spst.mannwhitneyu(learners_pre, non_learners_pre, alternative='two-sided')
    pval_pre = stat_test_pre.pvalue
    stat_test_post_3 = spst.mannwhitneyu(learners_post_3, non_learners_post_3, alternative='two-sided')
    pval_post_3 = stat_test_post_3.pvalue

    # Draw vertical annotation lines for both time points
    xloc_pre = -0.4
    learners_pre_mean = learners_pre.mean()
    non_learners_pre_mean = non_learners_pre.mean()
    axs.plot([xloc_pre, xloc_pre], [learners_pre_mean, non_learners_pre_mean], color='black', linestyle='-')
    axs.text(xloc_pre - 0.3, ((learners_pre_mean + non_learners_pre_mean) / 2),
             bpf.convert_pvalue_to_asterisks(pval_pre), ha='center', va='center', fontsize=12)

    xloc_post_3 = 1.4
    learners_post_3_mean = learners_post_3.mean()
    non_learners_post_3_mean = non_learners_post_3.mean()
    axs.plot([xloc_post_3, xloc_post_3], [learners_post_3_mean, non_learners_post_3_mean], color='black', linestyle='-')
    axs.text(xloc_post_3 + 0.3, ((learners_post_3_mean + non_learners_post_3_mean) / 2),
             bpf.convert_pvalue_to_asterisks(pval_post_3), ha='center', va='center', fontsize=12)

    # Set labels and adjust limits
    axs.set_ylabel(ylabel)
    axs.set_xlabel("time points\n(mins)")
    axs.set_xticklabels(["pre", "30 mins"])
    axs.set_ylim(ylim)
    axs.set_xlim(-0.75, 1.75)
    axs.spines[['right', 'top']].set_visible(False)

    return None


##plots with violin
#
#
#def plot_mini_feature(cells_df, field_to_plot, learners, non_learners, fig, axs):
#    # Set y-axis limits and labels based on the field to plot
#    if field_to_plot == "mepsp_amp":
#        ylim = (-1, 2)
#        ylabel = "mEPSP amplitude (mV)"
#    elif field_to_plot == "freq_mepsp":
#        ylim = (-1, 10)
#        ylabel = "mEPSP frequency (Hz)"
#    else:
#        ylim = (None, None)
#        ylabel = None
#
#    # Define the order of the time points
#    order = np.array(["pre", "post_3"])
#    cells_df = cells_df.copy()
#    data_to_plot = cells_df[cells_df["pre_post_status"].isin(order)]
#
#    # Function to filter outliers using the IQR rule
#    def remove_outliers(df, field):
#        q1 = df[field].quantile(0.25)
#        q3 = df[field].quantile(0.75)
#        iqr = q3 - q1
#        lower_bound = q1 - 1.5 * iqr
#        upper_bound = q3 + 1.5 * iqr
#        return df[(df[field] >= lower_bound) & (df[field] <= upper_bound)]
#
#    # Separate learners and non-learners and remove outliers
#    learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(learners)], field_to_plot).reset_index(drop=True)
#    non_learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(non_learners)], field_to_plot).reset_index(drop=True)
#
#    # Plot Violin Plots for learners and non-learners
#    sns.violinplot(data=learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                   order=order, color=bpf.CB_color_cycle[0], inner="quartile", linewidth=1)
#    sns.violinplot(data=non_learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                   order=order, color=bpf.CB_color_cycle[1], inner="quartile", linewidth=1)
#
#    # Make violins transparent
#    for violin in axs.collections:
#        violin.set_alpha(0.6)
#
#    # Annotate comparison within learners (pre vs. post_3)
#    stat_analysis = spst.mannwhitneyu(
#        learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot],
#        learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot],
#        alternative='two-sided'
#    )
#    pval_within_learners = stat_analysis.pvalue
#    annotator_within = Annotator(axs, [("pre", "post_3")], data=learners_df,
#                                 x="pre_post_status", y=field_to_plot, order=order)
#    annotator_within.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pval_within_learners)])
#    annotator_within.annotate()
#
#    # Perform comparisons between learners and non-learners at each time point
#    learners_pre = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot]
#    non_learners_pre = non_learners_df[non_learners_df["pre_post_status"] == "pre"][field_to_plot]
#    learners_post_3 = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot]
#    non_learners_post_3 = non_learners_df[non_learners_df["pre_post_status"] == "post_3"][field_to_plot]
#
#    # Perform Mann-Whitney U tests for both time points
#    stat_test_pre = spst.mannwhitneyu(learners_pre, non_learners_pre, alternative='two-sided')
#    pval_pre = stat_test_pre.pvalue
#    stat_test_post_3 = spst.mannwhitneyu(learners_post_3, non_learners_post_3, alternative='two-sided')
#    pval_post_3 = stat_test_post_3.pvalue
#
#    # Draw vertical annotation lines for both time points
#    xloc_pre = -0.4
#    learners_pre_mean = learners_pre.mean()
#    non_learners_pre_mean = non_learners_pre.mean()
#    axs.plot([xloc_pre, xloc_pre], [learners_pre_mean, non_learners_pre_mean], color='black', linestyle='-')
#    axs.text(xloc_pre - 0.1, ((learners_pre_mean + non_learners_pre_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_pre), ha='center', va='center', fontsize=12)
#
#    xloc_post_3 = 1.4
#    learners_post_3_mean = learners_post_3.mean()
#    non_learners_post_3_mean = non_learners_post_3.mean()
#    axs.plot([xloc_post_3, xloc_post_3], [learners_post_3_mean, non_learners_post_3_mean], color='black', linestyle='-')
#    axs.text(xloc_post_3 + 0.1, ((learners_post_3_mean + non_learners_post_3_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_post_3), ha='center', va='center', fontsize=12)
#
#    # Set labels and adjust limits
#    axs.set_ylabel(ylabel)
#    axs.set_xlabel("time points\n(mins)")
#    axs.set_xticklabels(["pre", "30 mins"])
#    axs.set_ylim(ylim)
#    axs.set_xlim(-0.75, 1.75)
#    axs.spines[['right', 'top']].set_visible(False)
#
#    return None

#def plot_mini_feature(cells_df, field_to_plot, learners, non_learners, fig, axs):
#    # Set y-axis limits and labels based on the field to plot
#    if field_to_plot == "mepsp_amp":
#        ylim = (-1, 2)
#        ylabel = "mEPSP amplitude (mV)"
#    elif field_to_plot == "freq_mepsp":
#        ylim = (-1, 10)
#        ylabel = "mEPSP frequency (Hz)"
#    else:
#        ylim = (None, None)
#        ylabel = None
#
#    # Define the order of the time points
#    order = np.array(["pre", "post_3"])
#    cells_df = cells_df.copy()
#    data_to_plot = cells_df[cells_df["pre_post_status"].isin(order)]
#    
#    # Function to filter outliers using the IQR rule
#    def remove_outliers(df, field):
#        q1 = df[field].quantile(0.25)
#        q3 = df[field].quantile(0.75)
#        iqr = q3 - q1
#        lower_bound = q1 - 1.5 * iqr
#        upper_bound = q3 + 1.5 * iqr
#        return df[(df[field] >= lower_bound) & (df[field] <= upper_bound)]
#
#    # Separate learners and non-learners and remove outliers
#    learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(learners)], field_to_plot).reset_index(drop=True)
#    non_learners_df = remove_outliers(data_to_plot[data_to_plot["cell_ID"].isin(non_learners)], field_to_plot).reset_index(drop=True)
#    
#    # Extract data for learners at each time point
#    pre_dat = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    post_dat = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#
#    # Use Mann-Whitney U test instead of Wilcoxon for non-paired data
#    stat_analysis = spst.mannwhitneyu(pre_dat, post_dat, alternative='two-sided')
#    pval_within_learners = stat_analysis.pvalue
#
#    # Plot Violin Plots for learners and non-learners without outliers
#    sns.violinplot(data=learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                   order=order, color=bpf.CB_color_cycle[0], alpha=0.6, inner="quartile", linewidth=1)
#    sns.violinplot(data=non_learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                   order=order, color=bpf.CB_color_cycle[1], alpha=0.6, inner="quartile", linewidth=1)
#
#    # Annotate comparison within learners (pre vs. post_3)
#    annotator_within = Annotator(axs, [("pre", "post_3")], data=learners_df,
#                                 x="pre_post_status", y=field_to_plot, order=order)
#    annotator_within.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pval_within_learners)])
#    annotator_within.annotate()
#
#    # Perform comparisons between learners and non-learners at each time point
#    learners_pre = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    non_learners_pre = non_learners_df[non_learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    learners_post_3 = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    non_learners_post_3 = non_learners_df[non_learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#
#    # Perform Mann-Whitney U tests for both time points
#    stat_test_pre = spst.mannwhitneyu(learners_pre, non_learners_pre, alternative='two-sided')
#    pval_pre = stat_test_pre.pvalue
#    
#    stat_test_post_3 = spst.mannwhitneyu(learners_post_3, non_learners_post_3, alternative='two-sided')
#    pval_post_3 = stat_test_post_3.pvalue
#
#    # Draw vertical annotation lines for both time points
#    xloc_pre = -0.4
#    learners_pre_mean = learners_pre.mean()
#    non_learners_pre_mean = non_learners_pre.mean()
#    axs.plot([xloc_pre, xloc_pre], [learners_pre_mean, non_learners_pre_mean], color='black', linestyle='-')
#    axs.text(xloc_pre - 0.1, ((learners_pre_mean + non_learners_pre_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_pre), ha='center', va='center', fontsize=12)
#    
#    xloc_post_3 = 1.4
#    learners_post_3_mean = learners_post_3.mean()
#    non_learners_post_3_mean = non_learners_post_3.mean()
#    axs.plot([xloc_post_3, xloc_post_3], [learners_post_3_mean, non_learners_post_3_mean], color='black', linestyle='-')
#    axs.text(xloc_post_3 + 0.1, ((learners_post_3_mean + non_learners_post_3_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_post_3), ha='center', va='center', fontsize=12)
#
#    # Set labels and adjust limits
#    axs.set_ylabel(ylabel)
#    axs.set_xlabel("time points\n(mins)")
#    axs.set_xticklabels(["pre", "30 mins"])
#    axs.set_ylim(ylim)
#    axs.set_xlim(-0.75, 1.75)
#    axs.spines[['right', 'top']].set_visible(False)
#
#    return None


#plots with only point plots
#def plot_mini_feature(cells_df, field_to_plot, learners, non_learners, fig, axs):
#    # Set y-axis limits and labels based on the field to plot
#    if field_to_plot == "mepsp_amp":
#        ylim = (-1, 2)
#        ylabel = "mEPSP amplitude (mV)"
#    elif field_to_plot == "freq_mepsp":
#        ylim = (-1, 10)
#        ylabel = "mEPSP frequency (Hz)"
#    else:
#        ylim = (None, None)
#        ylabel = None
#
#    # Define the order of the time points
#    order = np.array(["pre", "post_3"])
#    cells_df = cells_df.copy()
#    data_to_plot = cells_df[cells_df["pre_post_status"].isin(order)]
#    
#    # Separate learners and non-learners
#    learners_df = data_to_plot[data_to_plot["cell_ID"].isin(learners)].reset_index(drop=True)
#    non_learners_df = data_to_plot[data_to_plot["cell_ID"].isin(non_learners)].reset_index(drop=True)
#    
#    # Extract data for learners at each time point
#    pre_dat = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    post_dat = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    
#    # Plot point plots for learners and non-learners
#    sns.pointplot(data=learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                  order=order, color=bpf.CB_color_cycle[0], capsize=0.15, ci='sd')
#    sns.pointplot(data=non_learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                  order=order, color=bpf.CB_color_cycle[1], capsize=0.15, ci='sd')
#
#    # Statistical comparison within learners (pre vs. post_3)
#    stat_analysis = spst.wilcoxon(pre_dat, post_dat, zero_method="wilcox", correction=True)
#    pvalList = stat_analysis.pvalue
#    anotp_list = ["pre", "post_3"]
#    annotator = Annotator(axs, [anotp_list], data=learners_df,
#                          x="pre_post_status", y=field_to_plot, order=order)
#    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
#    annotator.annotate()
#
#    # New sections: Compare learners and non-learners at both "pre" and "post_3" time points
#    learners_pre = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    non_learners_pre = non_learners_df[non_learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    learners_post_3 = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    non_learners_post_3 = non_learners_df[non_learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    
#    # Perform Mann-Whitney U tests for both time points
#    stat_test_pre = spst.ttest_ind(learners_pre, non_learners_pre,
#                                   equal_var=False)
#    pval_pre = stat_test_pre.pvalue
#    
#    stat_test_post_3 = spst.ttest_ind(learners_post_3, non_learners_post_3,
#                                      equal_var=False)
#    pval_post_3 = stat_test_post_3.pvalue
#    
#    # Draw vertical annotation lines for both time points
#    # For "pre" time point (left side)
#    xloc_pre = -0.25  # Position the line slightly to the left of the "pre" mark
#    learners_pre_mean = learners_pre.mean()
#    non_learners_pre_mean = non_learners_pre.mean()
#    axs.plot([xloc_pre, xloc_pre], [learners_pre_mean, non_learners_pre_mean], color='black', linestyle='-')
#    axs.text(xloc_pre - 0.1, ((learners_pre_mean + non_learners_pre_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_pre), ha='center', va='center', fontsize=12)
#    
#    # For "post_3" (30 mins) time point (right side)
#    xloc_post_3 = 1.25  # Position the line slightly to the right of the "30 mins" mark
#    learners_post_3_mean = learners_post_3.mean()
#    non_learners_post_3_mean = non_learners_post_3.mean()
#    axs.plot([xloc_post_3, xloc_post_3], [learners_post_3_mean, non_learners_post_3_mean], color='black', linestyle='-')
#    axs.text(xloc_post_3 + 0.1, ((learners_post_3_mean + non_learners_post_3_mean) / 2),
#             bpf.convert_pvalue_to_asterisks(pval_post_3), ha='center', va='center', fontsize=12)
#
#    # Set labels and adjust limits
#    axs.set_ylabel(ylabel)
#    axs.set_xlabel("time points\n(mins)")
#    axs.set_xticklabels(["pre", "30 mins"])
#    axs.set_ylim(ylim)
#    axs.set_xlim(-0.5, 1.75)
#    axs.spines[['right', 'top']].set_visible(False)
#
#    return None

#def plot_mini_feature(cells_df, field_to_plot, learners, non_learners, fig, axs):
#    if field_to_plot == "mepsp_amp":
#        ylim = (-1, 2)
#        ylabel = ("mEPSP amplitude (mV)")
#    elif field_to_plot == "freq_mepsp":
#        ylim = (-1, 10)
#        ylabel = ("mEPSP frequency (Hz)")
#    else:
#        ylim = (None, None)
#        ylabel = None
#    order = np.array(["pre", "post_3"])
#    cells_df = cells_df.copy()
#    data_to_plot = cells_df[cells_df["pre_post_status"].isin(["pre", "post_3"])]
#    learners_df = data_to_plot[data_to_plot["cell_ID"].isin(learners)].reset_index(drop=True)
#    non_learners_df = data_to_plot[data_to_plot["cell_ID"].isin(non_learners)].reset_index(drop=True)
#    pre_dat = learners_df[learners_df["pre_post_status"] == "pre"][field_to_plot].reset_index(drop=True)
#    post_dat = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#
#    # Plot learners and non-learners
#    pointplot1 = sns.pointplot(data=learners_df, x="pre_post_status", y=field_to_plot, ax=axs,
#                               order=order, color=bpf.CB_color_cycle[0],
#                               capsize=0.15, ci='sd')
#    pointplot2 = sns.pointplot(data=non_learners_df, x="pre_post_status",
#                               y=field_to_plot, ax=axs, order=order,
#                               color=bpf.CB_color_cycle[1], capsize=0.15,
#                               ci='sd')
#
#    # Statistical comparison within learners (pre vs. post_3)
#    stat_analysis = spst.wilcoxon(pre_dat, post_dat, zero_method="wilcox", correction=True)
#    pvalList = stat_analysis.pvalue
#    anotp_list = ["pre", "post_3"]
#    annotator = Annotator(axs, [anotp_list], data=learners_df,
#                          x="pre_post_status", y=field_to_plot, order=order)
#    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
#    annotator.annotate()
#
#    # New section: Compare learners and non-learners at "post_3" (30 mins)
#    learners_post_3 = learners_df[learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    non_learners_post_3 = non_learners_df[non_learners_df["pre_post_status"] == "post_3"][field_to_plot].reset_index(drop=True)
#    
#    ## Perform Mann-Whitney U test or t-test between learners and non-learners at 30 mins
#    #stat_test_post_3 = spst.mannwhitneyu(learners_post_3, non_learners_post_3, alternative='two-sided')
#    #pval_post_3 = stat_test_post_3.pvalue
#
#    ## Manually draw a vertical line between learners and non-learners at "30 mins", shifted to the right
#    #xloc = 1.25  # Position the line slightly to the right of the "30 mins" mark
#    #learners_mean = learners_post_3.mean()
#    #non_learners_mean = non_learners_post_3.mean()
#    #axs.plot([xloc, xloc], [learners_mean*1.2, non_learners_mean*1.2], color='black', linestyle='-')
#
#    ## Add the p-value annotation manually
#    #axs.text(xloc + 0.1, ((learners_mean + non_learners_mean) / 2)-0.25, 
#    #         bpf.convert_pvalue_to_asterisks(pval_post_3),
#    #         ha='center', va='center', fontsize=12)
#
#    # Set labels and adjust limits
#    axs.set_ylabel(ylabel)
#    axs.set_xlabel("time points\n(mins)")
#    axs.set_xticklabels(["pre", "30 mins"])
#    axs.set_ylim(ylim)
#    
#    # Adjust the xlim to accommodate the moved annotation line
#    axs.set_xlim(-0.5, 1.75)  # Increased the upper limit to allow space for the annotation
#    
#    axs.spines[['right', 'top']].set_visible(False)
#    return None

















































































def plot_learner_vs_non_learner_mini_feature(cells_df,field_to_plot,learners,non_learners,fig,axs):
    if field_to_plot=="mepsp_amp":
        ylim=(-1,2)
    elif field_to_plot=="freq_mepsp":
        ylim=(-1,10)
    else:
        ylim=(None,None)
        ylabel=None
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
    #long_df['Values'] = pd.to_numeric(long_df['Values'], errors='coerce')


    sns.pointplot(data=long_df, x='Category', y='Values',ax=axs,
                 color=bpf.CB_color_cycle[3],capsize=0.15,ci='sd')
    #sns.stripplot(data=long_df, x='Category', y='Values',ax=axs,
    #             color=bpf.CB_color_cycle[3],alpha=0.1)

    #lrn_df = pd.DataFrame("learners":learners_dat,"non-learners":non_learners_dat})
    #
    #sns.pointplot(data=lrn_df,x="learners",y="non-learners",ax=axs,
    #              order=order,color=bpf.CB_color_cycle[0],
    #              errorbar='sd')
    stat_analysis= spst.f_oneway(learners_dat,non_learners_dat)
    pvalList = stat_analysis.pvalue
    anotp_list=["learners","non_learners"]
    annotator =Annotator(axs,[anotp_list], data=long_df,
                         x="Category",y="Values",order=order)
    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()
    axs.set_ylabel(None)
    axs.set_xlabel(None)
    axs.set_xticklabels(["learners","non-learners"])
    axs.set_yticklabels([])
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
    axs.set_ylim(ylim)
    axs.spines[['right', 'top']].set_visible(False)
    return None





def plot_mini_distribution(df_cells,dict_cell_classified,
                           fig, axs1,axs2):
    learners = dict_cell_classified["ap_cells"]["cell_ID"].unique()
    non_learners = dict_cell_classified["an_cells"]["cell_ID"].unique()
    
    norm_df_amp=df_cells.copy()
#    norm_df_amp = normalise_df_to_pre(norm_df_amp,"mepsp_amp")
    plot_mini_feature(norm_df_amp,"mepsp_amp",learners,non_learners,fig,axs1)
    norm_df_num = df_cells.copy()
#    norm_df_num = normalise_df_to_pre(norm_df_num,"num_mepsp")
#    plot_mini_feature(norm_df_num,"num_mepsp",learners,non_learners,fig,axs2)
    norm_df_freq = df_cells.copy()
#    norm_df_freq = normalise_df_to_pre(norm_df_freq,"freq_mepsp")
    plot_mini_feature(norm_df_freq,"freq_mepsp",learners,non_learners,fig,axs2)
    #plot_learner_vs_non_learner_mini_feature(df_cells,"mepsp_amp",
    #                                        learners,non_learners,fig,axs3)
    #plot_learner_vs_non_learner_mini_feature(df_cells,"freq_mepsp",
    #                                         learners,non_learners,fig,axs4)

#def plot_fi_curve(firing_properties,sc_data_dict,fig,axs):
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
#    sns.stripplot(data=firing_properties[firing_properties["cell_ID"].isin(learners)],
#                  x="injected_current",y="spike_frequency",
#                  alpha=0.2,color=bpf.CB_color_cycle[0])
#    sns.pointplot(data=firing_properties[firing_properties["cell_ID"].isin(learners)],
#                  x="injected_current",y="spike_frequency",color=bpf.CB_color_cycle[0], 
#                  capsize=0.15,label="learners")
#    sns.stripplot(data=firing_properties[firing_properties["cell_ID"].isin(non_learners)],
#                  x="injected_current",y="spike_frequency",
#                  alpha=0.2, color=bpf.CB_color_cycle[1])
#    sns.pointplot(data=firing_properties[firing_properties["cell_ID"].isin(non_learners)],
#                  x="injected_current",y="spike_frequency",color=bpf.CB_color_cycle[1],
#                  capsize=0.15, label="non-learners")
#    axs.set_ylabel("spike frequency\n(spikes/s)")
#    axs.set_xlabel("injected current\n(pA)")
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.xaxis.set_major_locator(MultipleLocator(6))
#    if axs.legend_ is not None:
#            axs.legend_.remove()
#


def plot_glm_curve(firing_properties, sc_data_dict, fig, axs):
    """
    Perform GLM, plot the predicted spike frequencies for learners and non-learners,
    overlay scatter points, and display R² (pseudo-R²) value.
    """
    # Extract learners and non-learners as before
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()

    # Assign 'cell_type' column based on learners and non-learners
    firing_properties["cell_type"] = firing_properties["cell_ID"].apply(
        lambda x: "learners" if x in learners else "non-learners" if x in non_learners else "unknown"
    )

    # Filter out "unknown" cell types (if any)
    filtered_df = firing_properties[firing_properties["cell_type"] != "unknown"]

    # ---- Perform GLM ---- #
    # Create a GLM formula: spike_frequency ~ injected_current + cell_type + (interaction term)
    formula = 'spike_frequency ~ injected_current + cell_type + injected_current:cell_type'
    
    # Fit a GLM (Gaussian family with an identity link function)
    glm_model = glm(formula=formula, data=filtered_df, family=sm.families.Gaussian()).fit()
    
    # Print the GLM summary (statistical significance of the terms)
    print(glm_model.summary())
    
    # Generate predicted values from the model for plotting
    filtered_df["glm_prediction"] = glm_model.fittedvalues

    # ---- Calculate R² (pseudo-R²) ---- #
    # Pseudo-R² (1 - (deviance/null_deviance))
    deviance = glm_model.deviance
    null_deviance = glm_model.null_deviance
    pseudo_r_squared = 1 - (deviance / null_deviance)

    # ---- Plot GLM predictions ---- #
    # Plot predicted spike frequencies for learners and non-learners
    sns.lineplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                 x="injected_current", y="glm_prediction", color=bpf.CB_color_cycle[0], 
                 label="GLM learners", ax=axs)
    sns.lineplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                 x="injected_current", y="glm_prediction", color=bpf.CB_color_cycle[1], 
                 label="GLM non-learners", ax=axs)

    # ---- Overlay scatter points ---- #
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                  x="injected_current", y="spike_frequency", alpha=0.3, color=bpf.CB_color_cycle[0], ax=axs)
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                  x="injected_current", y="spike_frequency", alpha=0.3, color=bpf.CB_color_cycle[1], ax=axs)

    # ---- Formatting the plot ---- #
    axs.set_ylabel("Spike Frequency / Predicted (spikes/s)")
    axs.set_xlabel("Injected Current (pA)")
    axs.spines[['right', 'top']].set_visible(False)
    axs.xaxis.set_major_locator(MultipleLocator(6))
    
    # Show the legend
    axs.legend(loc='upper left')

    # Annotate the R² value on the plot
    axs.annotate(f"Pseudo R²: {pseudo_r_squared:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return glm_model


def plot_fi_curve_with_ks(firing_properties, sc_data_dict, fig, axs):
    """
    Plot spike frequencies for learners and non-learners and perform KS test for each current level.
    """
    # Extract learners and non-learners as before
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()

    # Assign 'cell_type' column based on learners and non-learners
    firing_properties["cell_type"] = firing_properties["cell_ID"].apply(
        lambda x: "learners" if x in learners else "non-learners" if x in non_learners else "unknown"
    )

    # Filter out "unknown" cell types (if any)
    filtered_df = firing_properties[firing_properties["cell_type"] != "unknown"]

    # Plot as before
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                  x="injected_current", y="spike_frequency", alpha=0.2, color=bpf.CB_color_cycle[0])
    sns.pointplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                  x="injected_current", y="spike_frequency", color=bpf.CB_color_cycle[0], 
                  capsize=0.15, label="learners")
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                  x="injected_current", y="spike_frequency", alpha=0.2, color=bpf.CB_color_cycle[1])
    sns.pointplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                  x="injected_current", y="spike_frequency", color=bpf.CB_color_cycle[1], 
                  capsize=0.15, label="non-learners")

    # Formatting
    axs.set_ylabel("spike frequency\n(spikes/s)")
    axs.set_xlabel("injected current\n(pA)")
    axs.spines[['right', 'top']].set_visible(False)
    axs.xaxis.set_major_locator(MultipleLocator(6))
    if axs.legend_ is not None:
        axs.legend_.remove()

    # ---- Perform KS test for each current level ---- #
    p_values = []
    unique_currents = filtered_df["injected_current"].unique()

    for current in unique_currents:
        # Filter data for this specific current
        learners_current = filtered_df[(filtered_df["cell_type"] == "learners") & 
                                       (filtered_df["injected_current"] == current)]["spike_frequency"]
        non_learners_current = filtered_df[(filtered_df["cell_type"] == "non-learners") & 
                                           (filtered_df["injected_current"] == current)]["spike_frequency"]
        
        # Perform KS test (only if both groups have data)
        if len(learners_current) > 0 and len(non_learners_current) > 0:
            ks_stat, p_value = ks_2samp(learners_current, non_learners_current)
            p_values.append((current, p_value))
            print(f"KS test for {current} pA: p-value = {p_value:.3f}")
        else:
            p_values.append((current, np.nan))

    # ---- Overlay p-values on the plot ---- #
    # Annotate p-values for each injected current
    for current, p_value in p_values:
        if not np.isnan(p_value):
            axs.annotate(f"p={p_value:.3f}", xy=(current, axs.get_ylim()[1] * 0.95),
                         xytext=(0, 10), textcoords="offset points", ha='center', fontsize=10)

    return p_values



def plot_fi_curve_with_mixed_anova(firing_properties, sc_data_dict, fig, axs):
    """
    Plot spike frequencies for learners and non-learners and perform a Mixed ANOVA (within and between factors).
    Show significance on the plot.
    """
    # Extract learners and non-learners as before
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()

    # Assign 'cell_type' column based on learners and non-learners
    firing_properties["cell_type"] = firing_properties["cell_ID"].apply(
        lambda x: "learners" if x in learners else "non-learners" if x in non_learners else "unknown"
    )

    # Filter out "unknown" cell types (if any)
    filtered_df = firing_properties[firing_properties["cell_type"] != "unknown"]

    # ---- Perform Mixed ANOVA ---- #
    # Format data for mixed ANOVA
    aov_data = filtered_df.copy()
    
    # Rename columns to avoid issues with naming conventions
    aov_data = aov_data.rename(columns={"cell_ID": "Cell", "spike_frequency": "Spike_Frequency", "injected_current": "Current"})

    # Run Mixed ANOVA (Current as the within-subject factor, cell_type as the between-subject factor)
    aov = pg.mixed_anova(dv='Spike_Frequency', within='Current', between='cell_type', subject='Cell', data=aov_data)
    
    # Print the ANOVA results
    print(aov)
    
    # Determine significance from the ANOVA p-values
    current_p_value = aov.loc[aov['Source'] == 'Current', 'p-unc'].values[0]
    group_p_value = aov.loc[aov['Source'] == 'cell_type', 'p-unc'].values[0]
    interaction_p_value = aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0]

    # ---- Plot as before ---- #
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                  x="injected_current", y="spike_frequency", alpha=0.2, color=bpf.CB_color_cycle[0], ax=axs)
    sns.pointplot(data=filtered_df[filtered_df["cell_type"] == "learners"],
                  x="injected_current", y="spike_frequency", color=bpf.CB_color_cycle[0], 
                  capsize=0.15, label="learners", ax=axs)
    sns.stripplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                  x="injected_current", y="spike_frequency", alpha=0.2, color=bpf.CB_color_cycle[1], ax=axs)
    sns.pointplot(data=filtered_df[filtered_df["cell_type"] == "non-learners"],
                  x="injected_current", y="spike_frequency", color=bpf.CB_color_cycle[1], 
                  capsize=0.15, label="non-learners", ax=axs)

    # Formatting
    axs.set_ylabel("spike frequency\n(spikes/s)")
    axs.set_xlabel("injected current\n(pA)")
    axs.spines[['right', 'top']].set_visible(False)
    axs.xaxis.set_major_locator(MultipleLocator(6))
    if axs.legend_ is not None:
        axs.legend_.remove()

    # ---- Show significance from ANOVA results ---- #
    # Annotate significance on the plot
    if current_p_value < 0.05:
        axs.annotate(f"Current: p={current_p_value:.3f}*", xy=(0.7, 0.95), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        axs.annotate(f"Current: p={current_p_value:.3f}", xy=(0.7, 0.95), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if group_p_value < 0.05:
        axs.annotate(f"Group: p={group_p_value:.3f}*", xy=(0.7, 0.90), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        axs.annotate(f"Group: p={group_p_value:.3f}", xy=(0.7, 0.90), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if interaction_p_value < 0.05:
        axs.annotate(f"Interaction: p={interaction_p_value:.3f}*", xy=(0.7, 0.85), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        axs.annotate(f"Interaction: p={interaction_p_value:.3f}", xy=(0.7, 0.85), xycoords='axes fraction',
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return aov



def plot_fi_curve(firing_properties, sc_data_dict, fig, axs):
    """
    Plots the F-I curve for learners and non-learners with Mann-Whitney U test for each injected current.
    """
    # Identify learners and non-learners
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
    
    # Add a new column to indicate the group (learners or non-learners)
    firing_properties["Group"] = firing_properties["cell_ID"].apply(
        lambda x: "Learner" if x in learners else "Non-Learner"
    )
    
    # Separate data for learners and non-learners
    learners_data = firing_properties[firing_properties["cell_ID"].isin(learners)]
    non_learners_data = firing_properties[firing_properties["cell_ID"].isin(non_learners)]
    
    # Unique injected current values
    current_levels = firing_properties["injected_current"].unique()
    
    # Initialize dictionary to store significance results
    significance_dict = {}

    # Perform Mann-Whitney U test for each injected current level
    for current in current_levels:
        # Get spike frequencies for learners and non-learners at the current level
        learners_spikes = learners_data[learners_data["injected_current"] == current]["spike_frequency"]
        non_learners_spikes = non_learners_data[non_learners_data["injected_current"] == current]["spike_frequency"]
        
        # Perform Mann-Whitney U test
        if len(learners_spikes) > 0 and len(non_learners_spikes) > 0:
            stat, p_value = mannwhitneyu(learners_spikes, non_learners_spikes, alternative='two-sided')
        else:
            p_value = None
        
        # Store the p-value and convert to significance level
        if p_value is not None:
            significance = bpf.convert_pvalue_to_asterisks(p_value)
            if p_value > 0.05:
                significance = "ns"  # Not significant
        else:
            significance = "ns"
        
        significance_dict[current] = significance
    
    # Plot learners data with custom color
    sns.stripplot(data=learners_data, x="injected_current", y="spike_frequency",
                  alpha=0.2, color=bpf.CB_color_cycle[0])
    sns.pointplot(data=learners_data, x="injected_current", y="spike_frequency",
                  color=bpf.CB_color_cycle[0], capsize=0.15, label="learners")
    
    # Plot non-learners data with custom color
    sns.stripplot(data=non_learners_data, x="injected_current", y="spike_frequency",
                  alpha=0.2, color=bpf.CB_color_cycle[1])
    sns.pointplot(data=non_learners_data, x="injected_current", y="spike_frequency",
                  color=bpf.CB_color_cycle[1], capsize=0.15, label="non-learners")
    
    # Add labels and formatting
    axs.set_ylabel("spike frequency\n(spikes/s)")
    axs.set_xlabel("injected current\n(pA)")
    axs.spines[['right', 'top']].set_visible(False)
    axs.xaxis.set_major_locator(MultipleLocator(6))
    
    # Annotate with Mann-Whitney results for each current level
    #for current, significance in significance_dict.items():
    #    axs.text(current, 55, significance, ha='center')
    
    # Adjust legend if needed
    if axs.legend_ is not None:
        axs.legend_.remove()

    ## Optionally print significance results
    #print("Mann-Whitney U test results for each current level:")
    #for current, sig in significance_dict.items():
    #    print(f"Current {current}: {sig}")

#def plot_fi_curve(firing_properties, sc_data_dict, fig, axs):
#    """
#    Plots the F-I curve for learners and non-learners with statistical analysis using LME.
#    """
#    # Identify learners and non-learners
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
#    
#    # Add a new column to indicate the group (learners or non-learners)
#    firing_properties["Group"] = firing_properties["cell_ID"].apply(
#        lambda x: "Learner" if x in learners else "Non-Learner"
#    )
#    
#    # Separate data for learners and non-learners
#    learners_data = firing_properties[firing_properties["cell_ID"].isin(learners)]
#    non_learners_data = firing_properties[firing_properties["cell_ID"].isin(non_learners)]
#    
#    # Perform Linear Mixed-Effects Model using mixedlm
#    model = smf.mixedlm("spike_frequency ~ Group + injected_current", 
#                        data=firing_properties, 
#                        groups=firing_properties["cell_ID"])
#    result = model.fit()
#    print(f"LME: {result};;;;;;;;;;;;;;;;;;;;;;;;;;")
#    # Extract p-value for the effect of 'Group'
#    group_p_value = result.pvalues.get("Group[T.Learner]", None)
#    
#    # Convert p-value to your custom significance level annotation
#    if group_p_value is not None:
#        significance = bpf.convert_pvalue_to_asterisks(group_p_value)
#        if group_p_value > 0.05:
#            significance = "ns"  # Not significant
#    else:
#        significance = "ns"
#    
#    # Plot learners data with your custom color
#    sns.stripplot(data=learners_data, x="injected_current", y="spike_frequency",
#                  alpha=0.2, color=bpf.CB_color_cycle[0])
#    sns.pointplot(data=learners_data, x="injected_current", y="spike_frequency",
#                  color=bpf.CB_color_cycle[0], capsize=0.15, label="learners")
#    
#    # Plot non-learners data with your custom color
#    sns.stripplot(data=non_learners_data, x="injected_current", y="spike_frequency",
#                  alpha=0.2, color=bpf.CB_color_cycle[1])
#    sns.pointplot(data=non_learners_data, x="injected_current", y="spike_frequency",
#                  color=bpf.CB_color_cycle[1], capsize=0.15, label="non-learners")
#    
#    # Add labels and formatting
#    axs.set_ylabel("spike frequency\n(spikes/s)")
#    axs.set_xlabel("injected current\n(pA)")
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.xaxis.set_major_locator(MultipleLocator(6))
#    
#    # Annotate with overall significance from LME model without any styling
#    if significance:
#        axs.text(5, 50, f"LME: {significance}", ha='center')
#    
#    # Adjust legend if needed
#    if axs.legend_ is not None:
#        axs.legend_.remove()
#    
#    # Print LME summary for reference (optional)
#    print(result.summary())

#def plot_fi_curve(firing_properties, sc_data_dict, fig, axs):
#    #with man whitney test integrated
#    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
#    
#    # Separate data for learners and non-learners
#    learners_data = firing_properties[firing_properties["cell_ID"].isin(learners)]
#    non_learners_data = firing_properties[firing_properties["cell_ID"].isin(non_learners)]
#    
#    # Plotting learners
#    sns.stripplot(data=learners_data, x="injected_current", y="spike_frequency",
#                  alpha=0.2, color=bpf.CB_color_cycle[0])
#    sns.pointplot(data=learners_data, x="injected_current", y="spike_frequency", 
#                  color=bpf.CB_color_cycle[0], capsize=0.15, label="learners")
#    
#    # Plotting non-learners
#    sns.stripplot(data=non_learners_data, x="injected_current", y="spike_frequency",
#                  alpha=0.2, color=bpf.CB_color_cycle[1])
#    sns.pointplot(data=non_learners_data, x="injected_current", y="spike_frequency", 
#                  color=bpf.CB_color_cycle[1], capsize=0.15, label="non-learners")
#    
#    axs.set_ylabel("spike frequency\n(spikes/s)")
#    axs.set_xlabel("injected current\n(pA)")
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.xaxis.set_major_locator(MultipleLocator(6))
#
#    # Perform statistical tests on the spike frequency distributions
#    #ks_stat, ks_p_value = ks_2samp(learners_data["spike_frequency"], non_learners_data["spike_frequency"])
#    mw_stat, mw_p_value = mannwhitneyu(learners_data["spike_frequency"], non_learners_data["spike_frequency"])
#    annot_pval = bpf.convert_pvalue_to_asterisks(mw_p_value)
#    # Add statistical test results as text on the plot
#    #axs.text(0.05, 0.95, f'KS p-value: {ks_p_value:.3f}', transform=axs.transAxes, verticalalignment='top')
#    #axs.text(0.05, 0.90, f'{annot_pval}', transform=axs.transAxes, verticalalignment='top')
#    
#    # Adjust legend if needed
#    if axs.legend_ is not None:
#        axs.legend_.remove()
#


def plot_cell_distribution_plasticity(pd_cell_data_mean_cell_grp,
                                      fig,axs,cell_type):
    pd_cell_data_mean_pat_grp = pd_cell_data_mean_cell_grp.groupby(by='frame_status')
    axs.axvline(100,linestyle=":", color='k')
    axs.axhline(0.5,linestyle="--", color='r', label ="CDF =0.5")
    for f, frms in pd_cell_data_mean_pat_grp:
        if f == "point":
            continue
        else:
            pat_0_vals= frms[(frms["frame_id"]=="pattern_0")&(frms["pre_post_status"]=="post_3")]["max_trace %"].to_numpy()
            sns.kdeplot(data = pat_0_vals, cumulative = True,
                        label = "trained pattern", ax=axs,
                        color=bpf.CB_color_cycle[2],linewidth=3,
                        linestyle='-')
            pat_1_vals= frms[(frms["frame_id"]=="pattern_1")&(frms["pre_post_status"]=="post_3")]["max_trace %"].to_numpy()
            sns.kdeplot(data = pat_1_vals, cumulative = True,
                        label="overlapping\npattern", ax=axs,
                        color=bpf.CB_color_cycle[4],linewidth=3,
                        linestyle='--')
            pat_2_vals= frms[(frms["frame_id"]=="pattern_2")&(frms["pre_post_status"]=="post_3")]["max_trace %"].to_numpy()
            sns.kdeplot(data = pat_2_vals, cumulative = True, 
                        label="non-overlapping\npattern", ax=axs,
                        color=bpf.CB_color_cycle[5],linewidth=3,
                        linestyle=':')
    axs.set_xlim(-50,350)
    axs.set_title(cell_type)
    axs.set_xlabel("% change in response\nto patterns")
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor =(0.8, 0.275),
               ncol = 1,
               loc='upper center',frameon=False)#,loc='lower center'

    axs.spines[['right', 'top']].set_visible(False)
    if cell_type=="learners":
        axs.set_ylabel("CDF of cell numbers")
    else:
        axs.set_ylabel(None)
        axs.set_yticklabels([])
        pass
    #axs.xaxis.set_tick_params(labelsize=14,width=1.5,length=5)
    #axs.yaxis.set_tick_params(labelsize=14,width=1.5,length=5)
def norm_values(cell_list,val_to_plot):
    cell_list = cell_list.copy()
    cell_list = cell_list.copy()
    print(f"cell list inside func : {cell_list}")
    cell_grp=cell_list.groupby(by="cell_ID")
    for c, cell in cell_grp:
        pat_grp = cell.groupby(by="frame_id")
        for p,pat in pat_grp:
            if "pattern" not in p:
                continue
            else:
                #print(f"c:{c}, p:{p}")
                pre_val= float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]=="pre")][val_to_plot])
                pp_grp = pat.groupby(by="pre_post_status")
                for pr, pp in pp_grp:
                    norm_val = float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]==pr)][val_to_plot])
                    norm_val = (norm_val/pre_val)*100
                    cell_list.loc[(cell_list["cell_ID"]==c)&(cell_list["frame_id"]==p)&(cell_list["pre_post_status"]==pr),val_to_plot]=norm_val
    return cell_list
   
def plot_cell_dist(catcell_dist,val_to_plot,fig,axs,pattern_number,plt_color,
                  resp_color):
    y_lim = (-50,500)
    pat_num=int(pattern_number.split("_")[-1])
    num_cells= len(catcell_dist["cell_ID"].unique())
    pfd = catcell_dist.groupby(by="frame_id")
    for c, pat in pfd:
        if c != pattern_number:
            continue
        else:
            #pat = pat[(pat["pre_post_status"]!="post_5")]#&(pat["pre_post_status"]!="post_4")]#&(cell["pre_post_status"]!="post_3")]
            #order = np.array(('pre','post_0','post_1','post_2','post_3','post_4'),dtype=object)
            order = np.array(('pre','post_0','post_1','post_2','post_3'),dtype=object)
            #print(f"pat = &&&&&&&{pat}%%%%%%%%%%%%%")
            g=sns.stripplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
                            order=order,ax=axs,color=resp_color,
                            alpha=0.6,size=6, label='cell response')#alpha=0.8,
            sns.pointplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
                          errorbar="se",order=order,capsize=0.08,ax=axs,
                          color=plt_color, linestyles='dotted',scale = 0.5,
                         label="average cell response")
            #palette="pastel",hue="cell_ID")
            #g.legend_.remove()
            g.set_title(None)
            #"""
            pvalList = []
            anotp_list = []
            for i in order[1:]:
                posti ="post{i}"
                #non parametric, paired and small sample size, hence used Wilcoxon signed-rank test
                #Wilcoxon signed-rank test
                posti= spst.wilcoxon(pat[pat["pre_post_status"]=='pre'][f"{val_to_plot}"],pat[pat["pre_post_status"]==i][f"{val_to_plot}"],
                                     zero_method="wilcox", correction=True)
                pvalList.append(posti.pvalue)
                anotp_list.append(("pre",i))
            annotator = Annotator(axs,anotp_list,data=pat, 
                                  x="pre_post_status",
                                  y=f"{val_to_plot}",
                                  order=order,
                                 fontsize=8)
            #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
            #annotator.annotate()
            #"""
            axs.axhline(100, ls=':',color="k", alpha=0.4)
            if pat_num==0:
                sns.despine(fig=None, ax=axs, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs.set_ylabel("% change in\nEPSP amplitude")
                axs.set_xlabel(None)
                #axs[pat_num].set_yticks([])
            elif pat_num==1:
                sns.despine(fig=None, ax=axs, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs.set_ylabel(None)
            elif pat_num==2:
                sns.despine(fig=None, ax=axs, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs.set_xlabel(None)
                axs.set_ylabel(None)
            else:
                pass 
            g.set(ylim=y_lim)
            g.set_xticklabels(time_points,rotation=30)
            g.set_xlabel("time points (mins)")
            if g.legend_ is not None:
                g.legend_.remove()
    #axs.set_title("Cell distribution")
    ax_pos = axs.get_position()
    new_ax_pos = [ax_pos.x0-0.03, ax_pos.y0+0.005, ax_pos.width*1.1,
                  ax_pos.height*1.1]
    axs.set_position(new_ax_pos)



def plot_cell_category_classified_EPSP_peaks(pot_cells_df,dep_cells_df,
                                             val_to_plot,fig,axs):
    pot_cell_list= norm_values(pot_cells_df,val_to_plot)
    dep_cell_list= norm_values(dep_cells_df,val_to_plot)
    plot_cell_dist(pot_cell_list,val_to_plot,fig,axs,"pattern_0",
                   bpf.CB_color_cycle[0],bpf.CB_color_cycle[0])
    plot_cell_dist(dep_cell_list,val_to_plot,fig,axs,"pattern_0",
                   bpf.CB_color_cycle[1],bpf.CB_color_cycle[1])



def plot_cell_category_trace(fig, learner_status, gs, cell_df, label_letter, legend_added=False):
    sampling_rate = 20000  # for patterns
    sc_pat_grp = cell_df.groupby(by="frame_id")

    # Determine the color for the "post" trace based on learner status
    post_trace_color = bpf.CB_color_cycle[0] if learner_status == "learner" else bpf.CB_color_cycle[1]

    for pat, pat_data in sc_pat_grp:
        if "pattern" not in pat:
            continue
        else:
            pat_num = int(pat.split('_')[-1])
            pre_trace = pat_data[pat_data["pre_post_status"] == "pre"]["mean_trace"][0]
            post_trace = pat_data[pat_data["pre_post_status"] == "post_3"]["mean_trace"][0]
            
            # Determine the subplot location
            if learner_status == "learner":
                axs = fig.add_subplot(gs[pat_num, 4:6])
            else:
                axs = fig.add_subplot(gs[pat_num, 6:8])
            
            # Subtract baseline and truncate traces
            post_trace = bpf.substract_baseline(post_trace)
            post_trace = post_trace[:int(sampling_rate * time_to_plot)]
            pre_trace = bpf.substract_baseline(pre_trace)
            pre_trace = pre_trace[:int(sampling_rate * time_to_plot)]
            time = np.linspace(0, time_to_plot, len(post_trace)) * 1000
            
            # Plot the pre and post training traces with solid lines
            axs.plot(time, pre_trace, color=bpf.pre_color, label="pre training")
            axs.plot(time, post_trace, color=post_trace_color, label="post training")
            
            # Add the legend only once for learners
            if learner_status == "learner" and not legend_added:
                custom_handles = [
                    Line2D([0], [0], color=bpf.pre_color, linewidth=3, label="pre training"),
                    Line2D([0], [0], color=bpf.CB_color_cycle[0], linewidth=3,
                           label="learner\n(post training)"),
                    Line2D([0], [0], color=bpf.CB_color_cycle[1], linewidth=3,
                           label="non-learner\n(post training)")
                ]
                axs.legend(handles=custom_handles, 
                           loc='center', 
                           frameon=False, 
                           bbox_to_anchor=(-0.1,-3.155),
                           ncol=3)
                legend_added = True  # Ensure legend is added only once
            
            # Set axis labels and titles
            if pat_num == 0:
                axs.set_xlabel(None)
                axs.set_ylabel(None)
                axs.set_xticklabels([])
            elif pat_num == 1:
                axs.set_xlabel(None)
                axs.set_xticklabels([])
                axs.set_ylabel("cell response (mV)")
            elif pat_num == 2:
                axs.set_xlabel("time (ms)")
                axs.set_ylabel(None)
            
            if learner_status != 'learner':
                if pat_num == 0:
                    axs.set_title("non-learners")
                else:
                    axs.set_title(None)
                axs.set_yticklabels([])
                axs.set_ylabel(None)
            else:
                if pat_num == 0:
                    axs.set_title("learners")
                else:
                    axs.set_title(None)
            
            axs.set_ylim(-5, 6)
            axs.spines[['right', 'top']].set_visible(False)
    
    return legend_added


#def plot_cell_category_trace(fig, learner_status, gs, cell_df, label_letter):
#    sampling_rate = 20000  # for patterns
#    sc_pat_grp = cell_df.groupby(by="frame_id")
#    legend_added = False  # Flag to add legend only once
#
#    for pat, pat_data in sc_pat_grp:
#        if "pattern" not in pat:
#            continue
#        else:
#            pat_num = int(pat.split('_')[-1])
#            pre_trace = pat_data[pat_data["pre_post_status"] == "pre"]["mean_trace"][0]
#            post_trace = pat_data[pat_data["pre_post_status"] == "post_3"]["mean_trace"][0]
#            
#            # Determine the subplot location
#            if learner_status == "learner":
#                axs = fig.add_subplot(gs[pat_num, 4:6])
#            else:
#                axs = fig.add_subplot(gs[pat_num, 6:8])
#            
#            # Subtract baseline and truncate traces
#            post_trace = bpf.substract_baseline(post_trace)
#            post_trace = post_trace[:int(sampling_rate * time_to_plot)]
#            pre_trace = bpf.substract_baseline(pre_trace)
#            pre_trace = pre_trace[:int(sampling_rate * time_to_plot)]
#            time = np.linspace(0, time_to_plot, len(post_trace)) * 1000
#            
#            # Plot the pre and post training traces
#            axs.plot(time, pre_trace, color=bpf.pre_color, label="pre training trace")
#            axs.plot(time, post_trace, color=bpf.post_late, label="post training trace")
#            
#            # Add legend only for learners when pat_num == 2
#            if learner_status == "learner" and pat_num == 2 and not legend_added:
#                # Create custom legend handles with thicker lines
#                custom_handles = [
#                    Line2D([0], [0], color=bpf.pre_color, linewidth=3,
#                           label="pre training\nEPSP trace"),
#                    Line2D([0], [0], color=bpf.post_late, linewidth=3,
#                           label="post 30 mins of\ntraining EPSP trace")
#                ]
#                axs.legend(handles=custom_handles, 
#                           #bbox_to_anchor=(0.95, -0.75),
#                           bbox_to_anchor=(-0.6, -0.3),
#                           loc='center', frameon=False,
#                           ncol=1)
#                legend_added = True  # Ensure legend is added only once
#
#            # Axis labels and titles
#            if pat_num == 0:
#                axs.set_xlabel(None)
#                axs.set_ylabel(None)
#                axs.set_xticklabels([])
#            elif pat_num == 1:
#                axs.set_xlabel(None)
#                axs.set_xticklabels([])
#                axs.set_ylabel("cell response (mV)")
#            elif pat_num == 2:
#                axs.set_xlabel("time (ms)")
#                axs.set_ylabel(None)
#            
#            if learner_status != 'learner':
#                if pat_num == 0:
#                    axs.set_title("non-learners")
#                else:
#                    axs.set_title(None)
#                axs.set_yticklabels([])
#                axs.set_ylabel(None)
#            else:
#                if pat_num == 0:
#                    axs.set_title("learners")
#                else:
#                    axs.set_title(None)
#            
#            axs.set_ylim(-5, 6)
#            axs.spines[['right', 'top']].set_visible(False)

#def plot_cell_category_trace(fig,learner_status,gs,cell_df, label_letter):
#    sampling_rate = 20000 # for patterns
#    sc_pat_grp = cell_df.groupby(by="frame_id")
#    for pat, pat_data in sc_pat_grp:
#        if "pattern" not in pat:
#            continue
#        else:
#            pat_num = int(pat.split('_')[-1])
#            pre_trace  =pat_data[pat_data["pre_post_status"]=="pre"]["mean_trace"][0]
#            post_trace =pat_data[pat_data["pre_post_status"]=="post_3"]["mean_trace"][0]
#            print(f"pre_trace = {pre_trace}")
#            pps_grp = pat_data.groupby(by="pre_post_status")
#            print(f"pat num : {pat_num}, {pat}")
#            if learner_status=="learner":
#                axs = fig.add_subplot(gs[pat_num,4:6])
#            else:
#                axs = fig.add_subplot(gs[pat_num,6:8])
#            post_trace = bpf.substract_baseline(post_trace)
#            post_trace = post_trace[:int(sampling_rate*time_to_plot)]
#            pre_trace = bpf.substract_baseline(pre_trace)
#            pre_trace = pre_trace[:int(sampling_rate*time_to_plot)]
#            time = np.linspace(0,time_to_plot,len(post_trace))*1000
#            axs.plot(time,pre_trace, color=bpf.pre_color,
#                           label="pre training trace")
#            axs.plot(time,post_trace,
#                           color=bpf.post_late,
#                           label="post training trace")
#            #color=bpf.colorFader(bpf.post_color,
#            #                     bpf.post_late,
#            #                     (idx/len(pps_grp))))
#            if pat_num ==0:
#                axs.set_xlabel(None)
#                axs.set_ylabel(None)
#                axs.set_xticklabels([])
#            elif pat_num==1:
#                axs.set_xlabel(None)
#                axs.set_xticklabels([])
#                axs.set_ylabel("cell response (mV)")
#            elif pat_num==2:
#                axs.set_xlabel("time (ms)")
#                axs.set_ylabel(None)
#            else:
#                pass 
#            if learner_status!='learner':
#                if pat_num==0:
#                    axs.set_title("non-learners")
#                else:
#                    axs.set_title(None)
#                axs.set_yticklabels([])
#                axs.set_ylabel(None)
#            else:
#                if pat_num==0:
#                    axs.set_title("learners")
#                else:
#                    axs.set_title(None)
#            axs.set_ylim(-5,6)
#            axs.spines[['right', 'top']].set_visible(False)
#            #axs.text(-0.1,1.05,f'{label_letter}{pat_num+1}',transform=axs.transAxes,
#            #             fontsize=16, fontweight='bold', ha='center', va='center')

#with violin plots


def compare_cell_properties(cell_stats, fig, axs_rmp, axs_inr,
                            pot_cells_df, dep_cells_df):
    # Get the number of cells in each category
    num_pot_cells = f"no. of learners = {len(pot_cells_df['cell_ID'].unique())}"
    num_dep_cells = f"no. of non-learners = {len(dep_cells_df['cell_ID'].unique())}"
    
    cell_stat_with_category = []
    
    # Categorize cells into learners, non-learners, or feeble response
    for cell in cell_stats.iterrows():
        if cell[0] in list(pot_cells_df["cell_ID"]):
            cell_type = "learners"
            rmp = cell[1]["cell_stats"]["rmp_median"]
            inpR = cell[1]["cell_stats"]["InputR_cell_mean"]
        elif cell[0] in list(dep_cells_df["cell_ID"]):
            cell_type = "non-learners"
            rmp = cell[1]["cell_stats"]["rmp_median"]
            inpR = cell[1]["cell_stats"]["InputR_cell_mean"]
        else:
            cell_type = "feeble response"
            rmp = cell[1]["cell_stats"]["rmp_median"]
            inpR = cell[1]["cell_stats"]["InputR_cell_mean"]
        
        cell_stat_with_category.append([cell[0], cell_type, rmp, inpR])
    
    # Convert the list to a DataFrame
    c_cat_header = ["cell_ID", "cell_type", "rmp", "inpR"]
    cell_stat_with_category = pd.concat(pd.DataFrame([i], columns=c_cat_header) for i in cell_stat_with_category)
    
    # Filter out cells with feeble responses
    cell_stat_with_category = cell_stat_with_category[cell_stat_with_category["cell_type"] != "feeble response"]
    print(f"Cell stats with category: {cell_stat_with_category}")
    
    # Plot Resting Membrane Potential (RMP) with Violin Plot
    g1 = sns.violinplot(
        data=cell_stat_with_category,
        x="cell_type", y="rmp",
        ax=axs_inr, hue=None,
        palette={"learners": bpf.CB_color_cycle[0], "non-learners": bpf.CB_color_cycle[1]},
        inner="quartile", alpha=0.6
    )
    
    # Statistical test for RMP
    stat_testg1 = spst.mannwhitneyu(
        cell_stat_with_category[cell_stat_with_category["cell_type"] == "learners"]["rmp"],
        cell_stat_with_category[cell_stat_with_category["cell_type"] == "non-learners"]["rmp"],
        nan_policy='omit'
    )
    pvalLg1 = stat_testg1.pvalue

    # Filter out inpR values below 60 for plotting Input Resistance
    cell_stat_with_category_inpR_filtered = cell_stat_with_category[cell_stat_with_category["inpR"] > 60]

    # Check if the filtered DataFrame is not empty before plotting inpR
    if not cell_stat_with_category_inpR_filtered.empty:
        # Plot Input Resistance (InputR) with Violin Plot
        g2 = sns.violinplot(
            data=cell_stat_with_category_inpR_filtered,
            x="cell_type", y="inpR",
            ax=axs_rmp, hue=None,
            palette={"learners": bpf.CB_color_cycle[0], "non-learners": bpf.CB_color_cycle[1]},
            inner="quartile", alpha=0.6
        )
        
        # Statistical test for InputR (after filtering)
        stat_testg2 = spst.mannwhitneyu(
            cell_stat_with_category_inpR_filtered[cell_stat_with_category_inpR_filtered["cell_type"] == "learners"]["inpR"],
            cell_stat_with_category_inpR_filtered[cell_stat_with_category_inpR_filtered["cell_type"] == "non-learners"]["inpR"],
            nan_policy='omit'
        )
        pvalLg2 = stat_testg2.pvalue

        # Annotate InputR plot
        annotator2 = Annotator(axs_rmp, [("learners", "non-learners")],
                               data=cell_stat_with_category_inpR_filtered, x="cell_type", y="inpR")
        annotator2.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg2)])
        annotator2.annotate()
    else:
        print("No data available for Input Resistance plot (inpR > 60)")

    # Annotate RMP plot
    annotator1 = Annotator(axs_inr, [("learners", "non-learners")],
                           data=cell_stat_with_category, x="cell_type", y="rmp")
    annotator1.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg1)])
    annotator1.annotate()

    # Set limits and labels for the plots
    g1.set(xlim=(-0.5, 1.5), ylim=(-75, -60))
    if 'g2' in locals():
        g2.set(xlim=(-0.5, 1.5), ylim=(0, 250))
        g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
        g2.set_ylabel("Input Resistance\n(MOhms)")
        g2.set_xlabel(None)
        g2.legend_ = None
    
    g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
    g1.set_ylabel("Resting membrane\npotential (mV)")
    g1.set_xlabel(None)
    g1.legend_ = None
    
    # Remove spines for a cleaner look
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False)


##strip plot and point plot alone 
#def compare_cell_properties(cell_stats, fig,axs_rmp,axs_inr,
#                            pot_cells_df,dep_cells_df):
#    num_pot_cells =f"no. of learners = {len(pot_cells_df['cell_ID'].unique())}"
#    num_dep_cells =f"no. of non-learners = {len(dep_cells_df['cell_ID'].unique())}"
#    cell_stat_with_category=[]
#    for cell in cell_stats.iterrows():
#        if cell[0] in list(pot_cells_df["cell_ID"]):
#            cell_type =f"learners"
#            #keys in cell stats: ['InputR_cell_mean', 'inR_cut', 'rmp_ratio', 'rmp_cut_off', 'cell_status', 'inR_chancge', 'inR_cell', 'rmp_median']
#            rmp=cell[1]["cell_stats"]["rmp_median"]
#            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
#        elif cell[0] in list(dep_cells_df["cell_ID"]):
#            cell_type =f"non-learners"
#            #keys in cell stats: ['InputR_cell_mean', 'inR_cut', 'rmp_ratio', 'rmp_cut_off', 'cell_status', 'inR_chancge', 'inR_cell', 'rmp_median']
#            rmp=cell[1]["cell_stats"]["rmp_median"]
#            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
#        else:
#            cell_type = "feable response"
#            rmp=cell[1]["cell_stats"]["rmp_median"]
#            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
#        cell_stat_with_category.append([cell[0],cell_type,rmp,inpR])
#    c_cat_header=["cell_ID","cell_type","rmp","inpR"]
#    cell_stat_with_category =pd.concat(pd.DataFrame([i],columns=c_cat_header) for i in cell_stat_with_category)
#    cell_stat_with_category= cell_stat_with_category[cell_stat_with_category["cell_type"]!="feable response"]
#    print(f"cell stats with category:{cell_stat_with_category}")
#    g1=sns.stripplot(data=cell_stat_with_category,x="cell_type",y="rmp",ax=axs_inr, hue="cell_type",
#                       palette="colorblind",alpha=0.6,size=8)
#    sns.pointplot(data=cell_stat_with_category, x="cell_type",y=f"rmp",errorbar="se",
#                  capsize=0.15,ax=axs_inr,hue="cell_type", linestyles='')
#    #non parametric, unpaired, unequal sample size observations hence used kruskal
#    stat_testg1= spst.mannwhitneyu(cell_stat_with_category[(cell_stat_with_category["cell_type"]=="learners")]["rmp"],
#                             cell_stat_with_category[cell_stat_with_category["cell_type"]=="non-learners"]["rmp"],
#                             nan_policy='omit')
#    pvalLg1= stat_testg1.pvalue
#
#    g2=sns.stripplot(data=cell_stat_with_category,x="cell_type",y="inpR",ax=axs_rmp,hue="cell_type",
#                      palette="colorblind", alpha=0.6,size=8)
#    sns.pointplot(data=cell_stat_with_category, x="cell_type",y=f"inpR",errorbar="se",
#                  capsize=0.15,ax=axs_rmp,hue="cell_type", linestyles='')
#    stat_testg2= spst.mannwhitneyu(cell_stat_with_category[(cell_stat_with_category["cell_type"]=="learners")]["inpR"],
#                             cell_stat_with_category[cell_stat_with_category["cell_type"]=="non-learners"]["inpR"],
#                             nan_policy='omit')
#    pvalLg2= stat_testg2.pvalue
#    annotator1 = Annotator(axs_inr, [("learners","non-learners")],data=cell_stat_with_category, x="cell_type",y="rmp")
#    annotator1.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg1)])
#    annotator1.annotate()
#    annotator2 = Annotator(axs_rmp, [("learners","non-learners")],data=cell_stat_with_category, x="cell_type",y="inpR")
#    annotator2.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg2)])
#    annotator2.annotate()
#    
#    g1.set(xlim=(-1,2))
#    g1.set(ylim=(-75,-60))
#    g2.set(xlim=(-1,2))
#    g2.set(ylim=(50,200))
#    #g1.set_title("Resting membrane\npotential")
#    #g2.set_title("Input\nresistance")
#    g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
#    g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
#    g1.set_ylabel("Resting membrane\npotential(mV)")
#    g2.set_ylabel("Input Resistance\n(MOhms)")
#    g1.set_xlabel(None)
#    g2.set_xlabel(None)
#    handles, labels = axs_rmp.get_legend_handles_labels()
#    g1.legend_.remove()
#    g2.legend_.remove()
#    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)


def plot_figure_3(extracted_feature_pickle_file_path,
                  all_trails_all_Cells_path,
                  cell_categorised_pickle_file,
                  training_data_pickle_file,
                  firing_properties_path,
                  cell_stats_pickle_file,
                  illustration_path,
                  outdir,learner_cell=learner_cell,
                  non_learner_cell=non_learner_cell):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    all_trial_df = pd.read_pickle(all_trails_all_Cells_path)
    cell_stats_df = pd.read_hdf(cell_stats_pickle_file)
    training_data = pd.read_pickle(training_data_pickle_file)
    firing_properties= pd.read_pickle(firing_properties_path)
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
    illustration = pillow.Image.open(illustration_path)
    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 
                    0.8, 0.8, 1
                   ]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 
                     0.5, 0.5, 0.5, 0.5, 0.5,
                     #1, 1
                    ]       # Adjust these values as needed

    fig = plt.figure(figsize=(8,18))
    #gs = GridSpec(12, 8,width_ratios=width_ratios,
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.2)
    #place illustration
    axs_img = fig.add_subplot(gs[0:2, 0:2])
    plot_image(illustration,axs_img, 0,0,1)
    axs_img.text(0,1,'A',transform=axs_img.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')
    move_axis([axs_img],-0.08,0.0225,1.3)

    #plot EPSP classification for learner & non-learner
    learner_cell_df = learner_cell_df[~learner_cell_df["frame_status"].isin(deselect_list)]


    plot_cell_category_trace(fig,"learner",gs, learner_cell_df,"B")
    plot_cell_category_trace(fig,"non_learner",gs, non_learner_cell_df,"C")   
    
    axs_pat1 = fig.add_subplot(gs[0,2:3])
    axs_pat2 = fig.add_subplot(gs[1,2:3])
    axs_pat3 = fig.add_subplot(gs[2,2:3])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0.035,0,2)
    axs_pat1.text(0,1.35,'C',transform=axs_pat1.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')

    #plot distribution epsp for learners and non-leaners
    axs_dist1 = fig.add_subplot(gs[2:3,0:2])
    plot_cell_category_classified_EPSP_peaks(sc_data_dict["ap_cells"],
                                             sc_data_dict["an_cells"],
                                             "max_trace",fig,axs_dist1,
                                             )    
    axs_dist1.text(0.05,1,'B',transform=axs_dist1.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')
    move_axis([axs_dist1],0,0.05,1)
    
    #plot pie chart of the distribution
    axs_pie = fig.add_subplot(gs[4:6,0:2])
    plot_pie_cell_dis(fig,axs_pie,cell_dist,cell_dist_key)
    axs_pie.text(-0.025,0.85,'D',transform=axs_pie.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')
    move_axis([axs_pie],-0.075,0.115,1)
    
    #plot F-I curve
    axs_fi = fig.add_subplot(gs[4:6,0:2])
    #print(f"firing firing_properties: \n{firing_properties}")
    #plot_fi_curve_with_mixed_anova(firing_properties,sc_data_dict,fig,axs_fi)
    #plot_fi_curve_with_ks(firing_properties,sc_data_dict,fig,axs_fi)
    plot_fi_curve(firing_properties,sc_data_dict,fig,axs_fi)
    move_axis([axs_fi],-0.05,0,1)
    axs_fi.text(0.1,1,'E',transform=axs_fi.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')

    #plot cell property comparison
    axs_inr = fig.add_subplot(gs[4:6,3:5])
    axs_rmp = fig.add_subplot(gs[4:6,6:8])
    compare_cell_properties(cell_stats_df,fig,axs_inr,axs_rmp,
                            sc_data_dict["ap_cells"], sc_data_dict["an_cells"])
    move_axis([axs_inr],-0.045,0,1)
    axs_inr.text(0.1,1,'F',transform=axs_inr.transAxes,    
                fontsize=16, fontweight='bold', ha='center', va='center')
    axs_rmp.text(0.1,1,'G',transform=axs_rmp.transAxes,    
                fontsize=16, fontweight='bold', ha='center', va='center')
    
    

    #plot distribution of minis
    axs_mini_amp = fig.add_subplot(gs[7:8,0:2])
    #axs_mini_comp_amp = fig.add_subplot(gs[7:8,2:4])
    axs_mini_freq = fig.add_subplot(gs[7:8,2:4])
    #axs_mini_comp_freq = fig.add_subplot(gs[7:8,6:8])
    plot_mini_distribution(all_trial_df,sc_data_dict, fig, 
                           axs_mini_amp,axs_mini_freq,)
    move_axis([axs_mini_amp],-0.05,0.05,0.9)
    #move_axis([axs_mini_comp_amp],-0.05,0.05,0.9)
    move_axis([axs_mini_freq],0.05,0.05,0.9)
    #move_axis([axs_mini_comp_freq],0.05,0.05,0.9)
    axs_mini_list = [axs_mini_amp,axs_mini_freq]
    label_axis(axs_mini_list,"H")    
    
    #plot training timing details
    axs_trn = fig.add_subplot(gs[7:8,5:7])
    plot_threshold_timing(training_data,sc_data_dict,fig,axs_trn)
    move_axis([axs_trn],0.05,0.05,1)
    axs_trn.text(0.1,1,'I',transform=axs_trn.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')


    #plot CDF for cells
    axs_cdf1 = fig.add_subplot(gs[9:10,0:3])
    axs_cdf2 = fig.add_subplot(gs[9:10,3:6])
    plot_cell_distribution_plasticity(sc_data_dict["ap_cells"],
                                      fig,axs_cdf1,"learners")
    plot_cell_distribution_plasticity(sc_data_dict["an_cells"],
                                      fig,axs_cdf2,"non-learners")
    axs_cdf_list = [axs_cdf1,axs_cdf2]
    
    #label_axis(axs_cdf_list,"J")
    move_axis(axs_cdf_list,-0.05,0.075,1)
    label_axis(axs_cdf_list,"J")

    ##GLM on firing properties
    #axs_glm = fig.add_subplot(gs[10:11,0:2])
    #plot_glm_curve(firing_properties, sc_data_dict, fig, axs_glm)
    
    
    
    
    
    

    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_3.png"
    #outpath = f"{outdir}/figure_3.svg"
    #outpath = f"{outdir}/figure_3.pdf"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def main():
    # Argument parser.
    description = '''Generates figure 3'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pikl-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with extracted features'
                       )
    parser.add_argument('--alltrial-path', '-a'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with extracted features'
                       )
    parser.add_argument('--training-path', '-t'
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
                        , help = 'path to h5 file with cell stats'
                        'exrracted data'
                       )
    parser.add_argument('--firingproperties-path', '-q'
                        , required = False,default ='./', type=str
                        , help = 'path to h5 file with cell stats'
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
    alltrialspath=Path(args.alltrial_path)
    trainingpath = Path(args.training_path)
    scpath = Path(args.sortedcell_path)
    illustration_path = Path(args.illustration_path)
    cell_stat_path = Path(args.cellstat_path)
    firing_properties_path = Path(args.firingproperties_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_3'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_3(pklpath,alltrialspath,scpath,trainingpath,firing_properties_path,cell_stat_path,illustration_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
