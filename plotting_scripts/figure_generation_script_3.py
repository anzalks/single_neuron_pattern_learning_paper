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
        "X", "IX", "V", "iV",
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

def plot_threshold_timing(training_data,sc_data_dict,fig,axs):
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
    cell_grp = training_data.groupby(by="cell_ID")
    lrns=[]
    non_lrns=[]
    for cell, cell_data in cell_grp:
        if cell in learners:
            lrns.append(cell_data)
        elif cell in non_learners:
            non_lrns.append(cell_data)
        else:
            print(cell,"no selection")
            continue
    #print(lrns)
    #print(non_lrns)
    lrns = pd.concat(lrns)
    lrns["l_stat"]="learners"
    non_lrns = pd.concat(non_lrns)
    non_lrns["l_stat"]="non\nlearners"
    all_df = pd.concat([lrns,non_lrns])
    print(non_lrns["trigger_time"].unique())
    palette = {"learners": bpf.CB_color_cycle[0], "non\nlearners": bpf.CB_color_cycle[1]}
    sns.pointplot(data=all_df,x="l_stat",
                  y="cell_thresh_time",
                  hue="l_stat",
                  palette=palette,ci="sd",
                  capsize=0.15, dodge=True,ax=axs)
#    
#    sns.pointplot(data=all_df[all_df["l_stat"]=="non_learner"],x="l_stat",
#                  y="cell_thresh_time",
#                  color=bpf.CB_color_cycle[1],ci="sd" ,
#                  capsize=0.15,dodge=True,ax=axs)
#
    sns.stripplot(data=all_df,
                  x="l_stat", y="cell_thresh_time",
                  hue="l_stat",
                  palette=palette,alpha=0.5)
#    sns.stripplot(data=all_df[all_df["l_stat"]=="non_learner"] ,
#                  x="l_stat", y="cell_thresh_time",
#                  color=bpf.CB_color_cycle[1],alpha=0.5)
    axs.set_ylabel("rise time from \nprojection (ms)")
    axs.set_xlabel(None)
    axs.set_ylim(0,20)
    axs.spines[['right', 'top']].set_visible(False)
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(["learners","non-learners"], handles))
    fig.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor =(0.5, 0.31),
               ncol = 6,
               loc='upper center')#,frameon=False)#,loc='lower center'
    axs.legend_.remove()
def plot_mini_feature(cells_df,field_to_plot, learners,non_learners,fig,axs):
    if field_to_plot=="mepsp_amp":
        ylim=(-1,2)
        ylabel=("mEPSP amplitude (mV)")
    elif field_to_plot=="freq_mepsp":
        ylim=(-1,10)
        ylabel=("mEPSP frequency (Hz)")
    else:
        ylim=(None,None)
        ylabel=None
    order = np.array(["pre","post_3"])
    cells_df=cells_df.copy()
    data_to_plot = cells_df[cells_df["pre_post_status"].isin(["pre","post_3"])]
    learners_df =  data_to_plot[data_to_plot["cell_ID"].isin(learners)].reset_index(drop=True)
    non_learners_df =  data_to_plot[data_to_plot["cell_ID"].isin(non_learners)].reset_index(drop=True)
    pre_dat =  learners_df[learners_df["pre_post_status"]=="pre"][field_to_plot].reset_index(drop=True)
    post_dat =  learners_df[learners_df["pre_post_status"]=="post_3"][field_to_plot].reset_index(drop=True)

    pointplot1=sns.pointplot(data=learners_df,x="pre_post_status",y=field_to_plot,ax=axs,
                 order=order,color=bpf.CB_color_cycle[0],
                 capsize=0.15,ci='sd')
    pointplot2=sns.pointplot(data=non_learners_df,x="pre_post_status",
                  y=field_to_plot,ax=axs,order=order,
                  color=bpf.CB_color_cycle[1],capsize=0.15,
                  ci='sd')
    #sns.stripplot(data=learners_df,x="pre_post_status",y=field_to_plot,ax=axs,
    #             order=order,color=bpf.CB_color_cycle[0],
    #             alpha=0.1,zorder=1)
    #sns.stripplot(data=non_learners_df,x="pre_ost_status",
    #              y=field_to_plot,ax=axs,order=order,
    #              color=bpf.CB_color_cycle[1],alpha=0.1,zorder=1)


    stat_analysis= spst.wilcoxon(pre_dat,post_dat, zero_method="wilcox", correction=True)
    pvalList = stat_analysis.pvalue
    anotp_list=["pre","post_3"]
    annotator =Annotator(axs,[anotp_list], data=learners_df, 
                         x="pre_post_status",y=field_to_plot,order=order)
    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()
    axs.set_ylabel(ylabel)
    axs.set_xlabel("time points\n(mins)")
    axs.set_xticklabels(["pre","30 mins"])
    axs.set_ylim(ylim)
    axs.spines[['right', 'top']].set_visible(False)
    return None


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
                           fig, axs1,axs2,axs3,axs4):
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
    plot_learner_vs_non_learner_mini_feature(df_cells,"mepsp_amp",
                                            learners,non_learners,fig,axs3)
    plot_learner_vs_non_learner_mini_feature(df_cells,"freq_mepsp",
                                             learners,non_learners,fig,axs4)

def plot_fi_curve(firing_properties,sc_data_dict,fig,axs):
    learners = sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners = sc_data_dict["an_cells"]["cell_ID"].unique()
    sns.stripplot(data=firing_properties[firing_properties["cell_ID"].isin(learners)],
                  x="injected_current",y="spike_frequency",
                  alpha=0.2,color=bpf.CB_color_cycle[0])
    sns.pointplot(data=firing_properties[firing_properties["cell_ID"].isin(learners)],
                  x="injected_current",y="spike_frequency",color=bpf.CB_color_cycle[0], 
                  capsize=0.15,label="learners")
    sns.stripplot(data=firing_properties[firing_properties["cell_ID"].isin(non_learners)],
                  x="injected_current",y="spike_frequency",
                  alpha=0.2, color=bpf.CB_color_cycle[1])
    sns.pointplot(data=firing_properties[firing_properties["cell_ID"].isin(non_learners)],
                  x="injected_current",y="spike_frequency",color=bpf.CB_color_cycle[1],
                  capsize=0.15, label="non-learners")
    axs.set_ylabel("spike frequency\n(spikes/s)")
    axs.set_xlabel("injected current\n(pA)")
    axs.spines[['right', 'top']].set_visible(False)
    axs.xaxis.set_major_locator(MultipleLocator(6))
    if axs.legend_ is not None:
            axs.legend_.remove()
    

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
                        label="overlapping pattern", ax=axs,
                        color=bpf.CB_color_cycle[4],linewidth=3,
                        linestyle='--')
            pat_2_vals= frms[(frms["frame_id"]=="pattern_2")&(frms["pre_post_status"]=="post_3")]["max_trace %"].to_numpy()
            sns.kdeplot(data = pat_2_vals, cumulative = True, 
                        label="non-overlapping pattern", ax=axs,
                        color=bpf.CB_color_cycle[5],linewidth=3,
                        linestyle=':')
    axs.set_xlim(-50,350)
    axs.set_title(cell_type)
    axs.set_xlabel("% change in response\nto patterns")
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor =(0.5, 0.29),
               ncol = 6,
               loc='upper center')#,frameon=False)#,loc='lower center'

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


def plot_cell_category_trace(fig,learner_status,gs,cell_df, label_letter):
    sampling_rate = 20000 # for patterns
    sc_pat_grp = cell_df.groupby(by="frame_id")
    for pat, pat_data in sc_pat_grp:
        if "pattern" not in pat:
            continue
        else:
            pat_num = int(pat.split('_')[-1])
            pre_trace  =pat_data[pat_data["pre_post_status"]=="pre"]["mean_trace"][0]
            post_trace =pat_data[pat_data["pre_post_status"]=="post_3"]["mean_trace"][0]
            print(f"pre_trace = {pre_trace}")
            pps_grp = pat_data.groupby(by="pre_post_status")
            print(f"pat num : {pat_num}, {pat}")
            if learner_status=="learner":
                axs = fig.add_subplot(gs[pat_num,4:6])
            else:
                axs = fig.add_subplot(gs[pat_num,6:8])
            post_trace = bpf.substract_baseline(post_trace)
            post_trace = post_trace[:int(sampling_rate*time_to_plot)]
            pre_trace = bpf.substract_baseline(pre_trace)
            pre_trace = pre_trace[:int(sampling_rate*time_to_plot)]
            time = np.linspace(0,time_to_plot,len(post_trace))*1000
            axs.plot(time,pre_trace, color=bpf.pre_color,
                           label="pre training trace")
            axs.plot(time,post_trace,
                           color=bpf.post_late,
                           label="post training trace")
            #color=bpf.colorFader(bpf.post_color,
            #                     bpf.post_late,
            #                     (idx/len(pps_grp))))
            if pat_num ==0:
                axs.set_xlabel(None)
                axs.set_ylabel(None)
                axs.set_xticklabels([])
            elif pat_num==1:
                axs.set_xlabel(None)
                axs.set_xticklabels([])
                axs.set_ylabel("cell response (mV)")
            elif pat_num==2:
                axs.set_xlabel("time (ms)")
                axs.set_ylabel(None)
            else:
                pass 
            if learner_status!='learner':
                if pat_num==0:
                    axs.set_title("non-learners")
                else:
                    axs.set_title(None)
                axs.set_yticklabels([])
                axs.set_ylabel(None)
            else:
                if pat_num==0:
                    axs.set_title("learners")
                else:
                    axs.set_title(None)
            axs.set_ylim(-5,6)
            axs.spines[['right', 'top']].set_visible(False)
            #axs.text(-0.1,1.05,f'{label_letter}{pat_num+1}',transform=axs.transAxes,
            #             fontsize=16, fontweight='bold', ha='center', va='center')

def compare_cell_properties(cell_stats, fig,axs_rmp,axs_inr,
                            pot_cells_df,dep_cells_df):
    num_pot_cells =f"no. of learners = {len(pot_cells_df['cell_ID'].unique())}"
    num_dep_cells =f"no. of non-learners = {len(dep_cells_df['cell_ID'].unique())}"
    cell_stat_with_category=[]
    for cell in cell_stats.iterrows():
        if cell[0] in list(pot_cells_df["cell_ID"]):
            cell_type =f"learners"
            #keys in cell stats: ['InputR_cell_mean', 'inR_cut', 'rmp_ratio', 'rmp_cut_off', 'cell_status', 'inR_chancge', 'inR_cell', 'rmp_median']
            rmp=cell[1]["cell_stats"]["rmp_median"]
            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
        elif cell[0] in list(dep_cells_df["cell_ID"]):
            cell_type =f"non-learners"
            #keys in cell stats: ['InputR_cell_mean', 'inR_cut', 'rmp_ratio', 'rmp_cut_off', 'cell_status', 'inR_chancge', 'inR_cell', 'rmp_median']
            rmp=cell[1]["cell_stats"]["rmp_median"]
            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
        else:
            cell_type = "feable response"
            rmp=cell[1]["cell_stats"]["rmp_median"]
            inpR=cell[1]["cell_stats"]["InputR_cell_mean"]
        cell_stat_with_category.append([cell[0],cell_type,rmp,inpR])
    c_cat_header=["cell_ID","cell_type","rmp","inpR"]
    cell_stat_with_category =pd.concat(pd.DataFrame([i],columns=c_cat_header) for i in cell_stat_with_category)
    cell_stat_with_category= cell_stat_with_category[cell_stat_with_category["cell_type"]!="feable response"]
    print(f"cell stats with category:{cell_stat_with_category}")
    g1=sns.stripplot(data=cell_stat_with_category,x="cell_type",y="rmp",ax=axs_inr, hue="cell_type",
                       palette="colorblind",alpha=0.6,size=8)
    sns.pointplot(data=cell_stat_with_category, x="cell_type",y=f"rmp",errorbar="se",
                  capsize=0.15,ax=axs_inr,hue="cell_type", linestyles='')
    #non parametric, unpaired, unequal sample size observations hence used kruskal
    stat_testg1= spst.kruskal(cell_stat_with_category[(cell_stat_with_category["cell_type"]=="learners")]["rmp"],
                             cell_stat_with_category[cell_stat_with_category["cell_type"]=="non-learners"]["rmp"],
                             nan_policy='omit')
    pvalLg1= stat_testg1.pvalue

    g2=sns.stripplot(data=cell_stat_with_category,x="cell_type",y="inpR",ax=axs_rmp,hue="cell_type",
                      palette="colorblind", alpha=0.6,size=8)
    sns.pointplot(data=cell_stat_with_category, x="cell_type",y=f"inpR",errorbar="se",
                  capsize=0.15,ax=axs_rmp,hue="cell_type", linestyles='')
    stat_testg2= spst.kruskal(cell_stat_with_category[(cell_stat_with_category["cell_type"]=="learners")]["inpR"],
                             cell_stat_with_category[cell_stat_with_category["cell_type"]=="non-learners"]["inpR"],
                             nan_policy='omit')
    pvalLg2= stat_testg2.pvalue
    annotator1 = Annotator(axs_inr, [("learners","non-learners")],data=cell_stat_with_category, x="cell_type",y="rmp")
    annotator1.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg1)])
    annotator1.annotate()
    annotator2 = Annotator(axs_rmp, [("learners","non-learners")],data=cell_stat_with_category, x="cell_type",y="inpR")
    annotator2.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalLg2)])
    annotator2.annotate()
    
    g1.set(xlim=(-1,2))
    g1.set(ylim=(-75,-60))
    g2.set(xlim=(-1,2))
    g2.set(ylim=(50,200))
    #g1.set_title("Resting membrane\npotential")
    #g2.set_title("Input\nresistance")
    g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
    g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
    g1.set_ylabel("Resting membrane\npotential(mV)")
    g2.set_ylabel("Input Resistance\n(MOhms)")
    g1.set_xlabel(None)
    g2.set_xlabel(None)
    handles, labels = axs_rmp.get_legend_handles_labels()
    g1.legend_.remove()
    g2.legend_.remove()
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)


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
    width_ratios = [1, 1, 1, 1, 1, 0.8, 0.8,1]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]       # Adjust these values as needed

    fig = plt.figure(figsize=(8,18))
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.2)
    #place illustration
    axs_img = fig.add_subplot(gs[0:2, 0:2])
    plot_image(illustration,axs_img, 0,0,1)
    axs_img.text(0,1,'A',transform=axs_img.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')
    move_axis([axs_img],-0.075,0.025,1.3)

    #plot EPSP classification for learner & non-learner
    learner_cell_df = learner_cell_df[~learner_cell_df["frame_status"].isin(deselect_list)]


    plot_cell_category_trace(fig,"learner",gs, learner_cell_df,"B")
    plot_cell_category_trace(fig,"non_learner",gs, non_learner_cell_df,"C")   
    
    axs_pat1 = fig.add_subplot(gs[0,2:3])
    axs_pat2 = fig.add_subplot(gs[1,2:3])
    axs_pat3 = fig.add_subplot(gs[2,2:3])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0,0,2)
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
    axs_mini_comp_amp = fig.add_subplot(gs[7:8,2:4])
    axs_mini_freq = fig.add_subplot(gs[7:8,4:6])
    axs_mini_comp_freq = fig.add_subplot(gs[7:8,6:8])
    plot_mini_distribution(all_trial_df,sc_data_dict, fig, 
                           axs_mini_amp,axs_mini_freq,
                           axs_mini_comp_amp,axs_mini_comp_freq)
    move_axis([axs_mini_amp],-0.05,0.05,0.9)
    move_axis([axs_mini_comp_amp],-0.05,0.05,0.9)
    move_axis([axs_mini_freq],0.05,0.05,0.9)
    move_axis([axs_mini_comp_freq],0.05,0.05,0.9)
    axs_mini_list = [axs_mini_amp,axs_mini_comp_amp,
                     axs_mini_freq,axs_mini_comp_freq]
    label_axis(axs_mini_list,"H")    
    
    #plot CDF for cells
    axs_cdf1 = fig.add_subplot(gs[9:10,0:3])
    axs_cdf2 = fig.add_subplot(gs[9:10,3:6])
    plot_cell_distribution_plasticity(sc_data_dict["ap_cells"],
                                      fig,axs_cdf1,"learners")
    plot_cell_distribution_plasticity(sc_data_dict["an_cells"],
                                      fig,axs_cdf2,"non-learners")
    axs_cdf_list = [axs_cdf1,axs_cdf2]
    #label_axis(axs_cdf_list,"J")
    move_axis(axs_cdf_list,-0.05,0.05,1)
    label_axis(axs_cdf_list,"I")

    #plot training timing details
    axs_trn = fig.add_subplot(gs[9:10,6:8])
    plot_threshold_timing(training_data,sc_data_dict,fig,axs_trn)
    move_axis([axs_trn],0.05,0.05,1)
    axs_trn.text(0.1,1,'J',transform=axs_trn.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')

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
