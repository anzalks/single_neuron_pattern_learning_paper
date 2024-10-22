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
import baisic_plot_fuctnions_and_features as bpf

# plot features are defines in bpf
bpf.set_plot_properties()

vlinec = "#C35817"

learner_cell = "2022_12_21_cell_1" 
non_learner_cell = "2023_01_10_cell_1"
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

def plot_field_amplitudes_time_series(pd_cell_data_mean, trace_property,cell_type,axs1,axs2,axs3):
    pd_cell_data_mean = pd_cell_data_mean[pd_cell_data_mean["pre_post_status"]!="post_4"]
    order = np.array(('pre','post_0','post_1','post_2','post_3'),dtype=object)
    pd_cell_data_mean_cell_grp = pd_cell_data_mean.groupby(by='cell_ID')
    
    cells_ =[]
    for c, cell in pd_cell_data_mean_cell_grp:
        cell["min_f_norm"] = cell["min_field"]
        pat_grp = cell.groupby(by="frame_id")
        for pa, pat in pat_grp:
            if "pattern" in pa:
                #print(pa)
                pat_num = int(pa.split("_")[-1])
                pps = pat["pre_post_status"].unique()
                pre_minf_resp = float(pat[pat["pre_post_status"]=="pre"][trace_property])
                
                #print(f"pre_minf_resp={np.abs(pre_minf_resp)}")
                for p in pps:
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),trace_property] = np.abs(pat[pat["pre_post_status"]==f"{p}"][trace_property])/np.abs(pre_minf_resp)*100
                    field_raw = cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),trace_property]
                    #print(f'pps= {p}:::: field: {field_raw}')
                    pl_dat = cell[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}")]
                    replaced_ = cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),trace_property]
                #sns.stripplot(data=pat,x="pre_post_status", y= "min_f_norm", ax=axs[pat_num])
                
            else:
                continue
        cell=cell[cell["frame_status"]!="point"]
        cells_.append(cell)            

    pd_cell_data_mean= pd.concat(cells_)
    #sns.stripplot(data=pd_cell_data_mean,x="pre_post_status", y= "min_field", hue="cell_ID", alpha=0.5, palette="colorblind")
    patterns = pd_cell_data_mean["frame_id"].unique()
    cb_cyclno = [5,1,0]
    if cell_type=="learners":
        pltcolor = bpf.CB_color_cycle[0]
    else:
        pltcolor= bpf.CB_color_cycle[1]

    axslist = [axs1,axs2,axs3]
    learnt_pat_post_3_mean = pd_cell_data_mean[(pd_cell_data_mean["frame_id"]=="pattern_0")&(pd_cell_data_mean["pre_post_status"]=="post_3")][trace_property].mean()
    for pat_num in patterns:
        if "pattern" in pat_num:
            ax_no = int(pat_num.split("_")[-1])
            g= sns.stripplot(data=pd_cell_data_mean[pd_cell_data_mean["frame_id"]==pat_num],x="pre_post_status",
                          y= trace_property, alpha=0.8, color=bpf.CB_color_cycle[6], order=order, ax=axslist[ax_no],label=pat_num)
            sns.pointplot(data=pd_cell_data_mean[pd_cell_data_mean["frame_id"]==pat_num],x="pre_post_status",
                          y= trace_property,
                          color=pltcolor,errorbar="sd",capsize=0.1,
                          order=order, ax=axslist[ax_no], label=pat_num)

            pps_grp  = pd_cell_data_mean.groupby(by="pre_post_status")
            sns.despine(fig=None, ax=axslist[ax_no], top=True, right=True, left=False, bottom=False, offset=None, trim=False)

            pvalList = []
            anotp_list = []
            for i in order[1:]:
                posti ="post{i}"
                posti= spst.wilcoxon(pat[(pat["pre_post_status"]=='pre')&(pat["frame_id"]==pat_num)]["min_field"],pat[(pat["pre_post_status"]==i)&(pat["frame_id"]==pat_num)][trace_property],
                                     zero_method="wilcox", correction=True)
                pvalList.append(posti.pvalue)
                anotp_list.append(("pre",i))
            annotator = Annotator(axslist[ax_no],anotp_list,data=pat, x="pre_post_status",y=trace_property,order=order)
            #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
            #annotator.annotate()
            #"""
            axslist[ax_no].axhline(100,color='k', linestyle=':', alpha=0.4,linewidth=2)
            axslist[ax_no].axhline(learnt_pat_post_3_mean, color='r', linestyle='-.', alpha=0.5,linewidth=2)
            axslist[ax_no].set_xticklabels(["pre","0", "10", "20","30"])
            #axs[ax_no].axhline(125, color='r', linestyle=':', alpha=0.6,linewidth=3)



            handles, labels = axslist[ax_no].get_legend_handles_labels()
            if g.legend_ is not None:
                g.legend_.remove()
            if cell_type=="learners":
                g.set_xlabel(None)
                g.set_xticklabels([])
                if pat_num=="pattern_0":
                    g.set_ylabel("field response\n%change")

                else:
                    g.set_ylabel(None)
                    g.set_yticklabels([])
                    pass
                if pat_num=="pattern_1":
                    #g.set_title(cell_type)
                    g.set_xlabel(None)
                else:
                    g.set_title(None)
            else:
                if pat_num=="pattern_1":
                    g.set_xlabel("time points (mins)")
                    #g.set_title(cell_type)
                else:
                    #g.set_title(None)
                    g.set_xlabel(None)
                if pat_num=="pattern_0":
                    g.set_ylabel("field response\n%change")
                else:
                    g.set_ylabel(None)
                    g.set_yticklabels([])
            g.set_ylim(50,250)
        else:
            pass


def plot_raw_points(df_cells,pattern_num,field_to_plot,timepoint_to_plot, 
                    cell_type,fig, axs):
    df_cells=df_cells.copy()
    df_cells[field_to_plot] = df_cells[field_to_plot].abs()
    order = np.array(("pre",timepoint_to_plot),dtype=object)
    c_ratio = float(int(timepoint_to_plot.split("_")[-1])/4)
    
    pre_pat0 = np.array(df_cells[(df_cells["pre_post_status"]=="pre")&(df_cells["frame_id"]==pattern_num)][field_to_plot])
    post4_pat0 =np.array(df_cells[(df_cells["pre_post_status"]==timepoint_to_plot)&(df_cells["frame_id"]==pattern_num)][field_to_plot])
    
    x_pre=np.array(df_cells[(df_cells["pre_post_status"]=="pre")&(df_cells["frame_id"]==pattern_num)]["pre_post_status"])
    x_post=np.array(df_cells[(df_cells["pre_post_status"]==timepoint_to_plot)&(df_cells["frame_id"]==pattern_num)]["pre_post_status"])
    if cell_type=="learners":
        pltcolor = bpf.CB_color_cycle[0]
    else:
        pltcolor= bpf.CB_color_cycle[1]

    axs.scatter(x_pre,pre_pat0,color=bpf.CB_color_cycle[6],alpha=0.8)
    axs.scatter(x_post,post4_pat0,color=bpf.CB_color_cycle[6],alpha=0.8)
    for i in range(len(x_pre)):
        axs.plot([x_pre[i], x_post[i]], [pre_pat0[i], post4_pat0[i]], color=bpf.CB_color_cycle[6],alpha=0.8, linestyle='--')
    
    sns.pointplot(data=df_cells[(df_cells["pre_post_status"]==timepoint_to_plot)&(df_cells["frame_id"]==pattern_num)],
                  x="pre_post_status",y=field_to_plot,color=pltcolor,order=order,errorbar='sd',capsize=0.05,ax=axs,label="mean field response post training")
    sns.pointplot(data=df_cells[(df_cells["pre_post_status"]=="pre")&(df_cells["frame_id"]==pattern_num)],
                  x="pre_post_status",y=field_to_plot,color=bpf.pre_color,order=order,errorbar='sd',capsize=0.05,ax=axs,label="mean field response pre training")
    #axs.scatter(pre_pat0,post4_pat0)
    #axs.plot(pre_pat0,post4_pat0)

    pre= spst.wilcoxon(pre_pat0,post4_pat0,)# zero_method="wilcox", correction=True)

    pvalList=pre.pvalue
    #print(pvalList)
    anotp_list=("pre",timepoint_to_plot)
    annotator = Annotator(axs,[anotp_list],data=df_cells[(df_cells["frame_id"]==pattern_num)], x="pre_post_status",y=field_to_plot,order=order)

    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()
    sns.despine(fig=None, ax=axs, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    
    #put strip plot on top of the points
    plt.setp(axs.lines, zorder=100)
    plt.setp(axs.collections, zorder=100, label="")    
    #"""
    #plt.ylim(-3,0.5)
    #axs.get_legend().set_visible(False)
    #plt.setp(axs.spines.values(), linewidth=1.5)
    #axs.xaxis.set_tick_params(labelsize=14,width=1.5,length=5)
    #axs.yaxis.set_tick_params(labelsize=14,width=1.5,length=5)
    axs.set_xticklabels(["pre","30"])
    axs.set_xlim(-0.25,1.5)
    axs.set_ylim(-0.25,3)
    if pattern_num=="pattern_0":
        axs.set_ylabel("Field response\n(mV)", fontsize=14)
    else:
        axs.set_ylabel(None)
        axs.set_yticklabels([])
    if pattern_num=="pattern_1":
        axs.set_xlabel("time points (mins)")
    else:
        axs.set_xlabel(None)
    
def plot_field_response_pairs(df_cells,feature,timepoint,cell_grp_type,nomr_status,fig, axs1,axs2,axs3):
    plot_raw_points(df_cells,"pattern_0",feature,timepoint, cell_grp_type, fig, axs1)
    plot_raw_points(df_cells,"pattern_1",feature,timepoint, cell_grp_type, fig, axs2)
    plot_raw_points(df_cells,"pattern_2",feature,timepoint, cell_grp_type, fig, axs3)

#def plot_minf_compare_all_pat(df_cells,sc_data_dict,fig,axs):
#    learners=sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners=sc_data_dict["an_cells"]["cell_ID"].unique()
#    cell_grp= df_cells.groupby(by="cell_ID")
#    order=np.array(["post_3"])
#    cells_ =[]
#    for c, cell in cell_grp:
#        cell["min_f_norm"] = cell["min_field"]
#        pat_grp = cell.groupby(by="frame_id")
#        for pa, pat in pat_grp:
#            if "pattern" in pa:
#                #print(pa)
#                pat_num = int(pa.split("_")[-1])
#                pps = pat["pre_post_status"].unique()
#                pre_minf_resp = float(pat[pat["pre_post_status"]=="pre"]["min_field"])
#                #print(f"pre_minf_resp={np.abs(pre_minf_resp)}")
#                for p in pps:
#                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_field"]= np.abs(pat[pat["pre_post_status"]==f"{p}"]["min_field"])/np.abs(pre_minf_resp)*100
#                    field_raw = cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_field"]
#                    #print(f'pps= {p}:::: field: {field_raw}')
#                    pl_dat = cell[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}")]
#                    replaced_ =  cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_field"]
#            else:
#                continue
#            cell=cell[cell["frame_status"]!="point"]
#            cells_.append(cell)            
#    all_cell_df= pd.concat(cells_)
#    cell_grp= all_cell_df.groupby(by="cell_ID") 
#    learners_all_pat =  all_cell_df[(all_cell_df["cell_ID"].isin(learners))&(all_cell_df["pre_post_status"]=="post_3")]
#    print(learners_all_pat)
#    non_learners_all_pat = all_cell_df[(all_cell_df["cell_ID"].isin(non_learners))&(all_cell_df["pre_post_status"]=="post_3")]
#    sns.pointplot(data=learners_all_pat,x="frame_id",y="min_field",
#                  color=bpf.CB_color_cycle[0], ax=axs,errorbar='sd',
#                  capsize=0.15)
#    sns.pointplot(data=non_learners_all_pat,x="frame_id",y="min_field",
#                  color=bpf.CB_color_cycle[1], ax=axs,errorbar='sd',
#                  capsize=0.15)
#    axs.spines[['right', 'top']].set_visible(False)
#    axs.set_xticklabels(["trained\npattern","overlapping\npattern",
#                        "non-overlapping\npattern"
#                        ],rotation=30)
#    axs.set_ylabel("% change in field")
#    axs.set_xlabel("patterns")



def plot_minf_compare_all_pat(df_cells, sc_data_dict, fig, axs):
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
    cell_grp = df_cells.groupby(by="cell_ID")
    order = np.array(["post_3"])
    cells_ = []

    # Process the data and normalize 'min_field' values
    for c, cell in cell_grp:
        cell["min_f_norm"] = cell["min_field"]
        pat_grp = cell.groupby(by="frame_id")
        for pa, pat in pat_grp:
            if "pattern" in pa:
                pat_num = int(pa.split("_")[-1])
                pps = pat["pre_post_status"].unique()
                pre_minf_resp = float(pat[pat["pre_post_status"] == "pre"]["min_field"])
                for p in pps:
                    cell.loc[(cell["frame_id"] == f"{pa}") & (cell["pre_post_status"] == f"{p}"), "min_field"] = (
                        np.abs(pat[pat["pre_post_status"] == f"{p}"]["min_field"]) / np.abs(pre_minf_resp) * 100
                    )
            else:
                continue
            cell = cell[cell["frame_status"] != "point"]
            cells_.append(cell)

    # Combine all processed cells
    all_cell_df = pd.concat(cells_)
    cell_grp = all_cell_df.groupby(by="cell_ID")
    
    # Separate learners and non-learners for 'post_3'
    learners_all_pat = all_cell_df[(all_cell_df["cell_ID"].isin(learners)) & (all_cell_df["pre_post_status"] == "post_3")]
    non_learners_all_pat = all_cell_df[(all_cell_df["cell_ID"].isin(non_learners)) & (all_cell_df["pre_post_status"] == "post_3")]

    # Plot learners and non-learners
    sns.pointplot(data=learners_all_pat, x="frame_id", y="min_field",
                  color=bpf.CB_color_cycle[0], ax=axs, errorbar='sd', capsize=0.15)
    sns.pointplot(data=non_learners_all_pat, x="frame_id", y="min_field",
                  color=bpf.CB_color_cycle[1], ax=axs, errorbar='sd', capsize=0.15)

    # Set x and y labels and customize the plot
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xticklabels(["trained\npattern", "overlapping\npattern", "non-overlapping\npattern"], rotation=30)
    axs.set_ylabel("% change in field")
    axs.set_xlabel("patterns")

    # Check the actual values in the "frame_id" column
    print(all_cell_df["frame_id"].unique())  # This will show the actual values

    # Perform Mann-Whitney U test for each pattern
    patterns = all_cell_df["frame_id"].unique()  # Use actual unique pattern names from data
    p_values = []
    for pattern in patterns:
        learners_data = learners_all_pat[learners_all_pat["frame_id"] == pattern]["min_field"]
        non_learners_data = non_learners_all_pat[non_learners_all_pat["frame_id"] == pattern]["min_field"]
        stat_test = spst.mannwhitneyu(learners_data, non_learners_data, alternative='two-sided')
        p_values.append(stat_test.pvalue)

    # Create a custom Annotator for the significance levels
    annotator = Annotator(axs, [(patterns[0], patterns[0]),
                                (patterns[1], patterns[1]),
                                (patterns[2], patterns[2])],
                          data=all_cell_df, x="frame_id", y="min_field")
    
    # Set custom annotations using p-value converted to asterisks and annotate
    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(p) for p in p_values])
    annotator.annotate()






































































def plot_figure_6(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  cell_stats_pickle_file,
                  outdir,learner_cell=learner_cell,
                  non_learner_cell=non_learner_cell):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    cell_stats_df = pd.read_hdf(cell_stats_pickle_file)
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
                     1, 1
                    ]
                     # Adjust these values as needed
    
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1
                   ]# Adjust these values as needed

    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(12, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.5, hspace=0.5)



    #plot patterns
    axs_pat_1 = fig.add_subplot(gs[0:1,0:1])
    axs_pat_2 = fig.add_subplot(gs[0:1,1:2])
    axs_pat_3 = fig.add_subplot(gs[0:1,2:3])
    axs_pat_4 = fig.add_subplot(gs[0:1,4:5])
    axs_pat_5 = fig.add_subplot(gs[0:1,5:6])
    axs_pat_6 = fig.add_subplot(gs[0:1,6:7])
    plot_patterns(axs_pat_1,axs_pat_2,axs_pat_3,0,0,2)
    plot_patterns(axs_pat_4,axs_pat_5,axs_pat_6,-0.05,0,2)


    #plot distribution epsp for learners and leaners
    axs_ex_pat1 = fig.add_subplot(gs[1:3,0:1])
    axs_ex_pat2 = fig.add_subplot(gs[1:3,1:2])
    axs_ex_pat3 = fig.add_subplot(gs[1:3,2:3])
    plot_field_response_pairs(sc_data_dict["ap_cells"],"min_field","post_3",
                              "learners","no norm",
                              fig,axs_ex_pat1,axs_ex_pat2,axs_ex_pat3)
    axs_ex_fl_list = [axs_ex_pat1,axs_ex_pat2,axs_ex_pat3]
    label_axis(axs_ex_fl_list,"A")
    move_axis(axs_ex_fl_list,0,0,1)

    axs_in_pat1 = fig.add_subplot(gs[1:3,4:5])
    axs_in_pat2 = fig.add_subplot(gs[1:3,5:6])
    axs_in_pat3 = fig.add_subplot(gs[1:3,6:7])
    plot_field_response_pairs(sc_data_dict["an_cells"],"min_field","post_3",
                              "non-learners","no norm",
                              fig,axs_in_pat1,axs_in_pat2,axs_in_pat3)
    
    axs_in_fl_list = [axs_in_pat1,axs_in_pat2,axs_in_pat3]
    label_axis(axs_in_fl_list,"B")
    move_axis(axs_in_fl_list,-0.05,0,1)
    
    axs_pat_fl1 = fig.add_subplot(gs[4:5,0:1])
    axs_pat_fl2 = fig.add_subplot(gs[4:5,2:3])
    axs_pat_fl3 = fig.add_subplot(gs[4:5,4:5])
    plot_patterns(axs_pat_fl1,axs_pat_fl2,axs_pat_fl3,0.07,0,1)
    axs_pat_list = [axs_pat_fl1,axs_pat_fl2,axs_pat_fl3]
    move_axis(axs_pat_list,-0.02,0,1)


    axs_ex_fl1 = fig.add_subplot(gs[5:7,0:2])
    axs_ex_fl2 = fig.add_subplot(gs[5:7,2:4])
    axs_ex_fl3 = fig.add_subplot(gs[5:7,4:6])
    plot_field_amplitudes_time_series(sc_data_dict["ap_cells"],"min_field",
                                      "learners",axs_ex_fl1,axs_ex_fl2,axs_ex_fl3)
    axs_ex_fl_list = [axs_ex_fl1,axs_ex_fl2,axs_ex_fl3]
    label_axis(axs_ex_fl_list,"C")    
    axs_in_fl1 = fig.add_subplot(gs[7:9,0:2])
    axs_in_fl2 = fig.add_subplot(gs[7:9,2:4])
    axs_in_fl3 = fig.add_subplot(gs[7:9,4:6])
    plot_field_amplitudes_time_series(sc_data_dict["an_cells"],"min_field",
                                      "non-learners",axs_in_fl1,axs_in_fl2,axs_in_fl3)
    axs_in_fl_list = [axs_in_fl1,axs_in_fl2,axs_in_fl3]
    label_axis(axs_in_fl_list,"D")
    axs_slope = fig.add_subplot(gs[9:11,0:2])
    plot_minf_compare_all_pat(feature_extracted_data,sc_data_dict,fig,axs_slope)
    move_axis([axs_slope],0,-0.05,1)
    axs_slope.text(0.05,1,'E',transform=axs_slope.transAxes,    
                        fontsize=16, fontweight='bold',
                        ha='center',va='center')

    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_6.png"
    #outpath = f"{outdir}/figure_6.svg"
    #outpath = f"{outdir}/figure_6.pdf"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser.
    description = '''Generates figure 6'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pikl-path', '-f'
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
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_6'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_6(pklpath,scpath,cell_stat_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
