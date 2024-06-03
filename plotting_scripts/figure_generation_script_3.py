__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 1 of pattern learning paper.
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

time_points = ["pre","0 mins", "10 mins", "20 mins","30 mins" ]
selected_time_points = ['post_0', 'post_1', 'post_2', 'post_3','pre']
                        #'post_4','post_5']

class Args: pass
args_ = Args()

def plot_cell_category_trace(fig,learner_status,gs,cell_df):
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
                continue
            if learner_status!='learner':
                axs.set_yticklabels([])
                axs.set_ylabel(None)
            axs.set_ylim(-5,6)
            axs.spines[['right', 'top']].set_visible(False)

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
    
    axs_inr.text(-0.5,1.4,'C',transform=axs_inr.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')    
    g1.set(xlim=(-1,2))
    g1.set(ylim=(-75,-60))
    g2.set(xlim=(-1,2))
    g2.set(ylim=(50,200))
    g1.set_title("Resting membrane\npotential")
    g2.set_title("Input\nresistance")
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


def plot_figure_2(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  cell_stats_pickle_file,
                  illustration_path,
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
    illustration = pillow.Image.open(illustration_path)
    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 1, 1,1]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5]       # Adjust these values as needed

    fig = plt.figure(figsize=(8,18))
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.3)
    #place illustration
    ax_img = fig.add_subplot(gs[0:2, 0:2])
    ax_img.imshow(illustration)

    ax_img.axis('off')
    ax_img.text(0.1,2.2,'A',transform=ax_img.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')


    #plot EPSP classification for learner & non-learner
    learner_cell_df = learner_cell_df[~learner_cell_df["frame_status"].isin(deselect_list)]


    plot_cell_category_trace(fig,"learner",gs, learner_cell_df)
    plot_cell_category_trace(fig,"non_learner",gs, non_learner_cell_df)   
        
    pattern_list = ["trained\npattern","Overlapping\npattern",
                    "Non-overlapping\npattern"]
    for pr_no, pattern in enumerate(pattern_list):
        if pr_no==0:
            axs_pat = fig.add_subplot(gs[pr_no,2:3])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(0,2)
            axs_pat.imshow(pat_fr)
            axs_pat.text(0.1,2.5,'B',transform=axs_pat.transAxes,    
                        fontsize=16, fontweight='bold', ha='center', va='center')
        elif pr_no==1:
            axs_pat = fig.add_subplot(gs[pr_no,2:3])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,2)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = fig.add_subplot(gs[pr_no,2:3])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,2)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        #pat_pos = axs_pat.get_position()
        #new_pat_pos = [pat_pos.x0-0.07, pat_pos.y0, pat_pos.width,
        #                pat_pos.height]
        #axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        #axs_pat.set_title(pattern)
    axs_inr = fig.add_subplot(gs[4:6,3:5])
    axs_rmp = fig.add_subplot(gs[4:6,6:8])
    compare_cell_properties(cell_stats_df,fig,axs_rmp,axs_inr,
                            sc_data_dict["ap_cells"], sc_data_dict["an_cells"])



    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_3.png"
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
    globoutdir= globoutdir/'Figure_3'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_2(pklpath,scpath,cell_stat_path,illustration_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
