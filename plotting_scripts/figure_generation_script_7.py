__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 7 of pattern learning paper.
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
    
def label_axis(axis_list,letter_label):
    for axs_no, axs in enumerate(axis_list):
        axs_no = axs_no+1
        axs.text(-0.075,1.1,f'{letter_label}{axs_no}',transform=axs.transAxes,    
                      fontsize=16, fontweight='bold', ha='center', va='center')

#def gama_fit(expt,alp,bet,gam):
#    return expt-(((bet*expt)/(gam+expt))*expt)-alp

def gama_fit(expt,alp,bet,gam):
    bet=1
    alp=0
    return expt-(((bet*expt)/(gam+expt))*expt)-alp


def eq_fit(list_of_x_y_responses_pre,list_of_x_y_responses,pat_num,eq_pre_max,
           eq_post_max,cell_type,fig, axs):
    x_y_responses = np.array(list_of_x_y_responses)
    pre_x_y = np.array(list_of_x_y_responses_pre)
    pre_arr1,pre_arr2=np.split(pre_x_y,2,axis=1)
    pre_arr1 =np.ravel(pre_arr1)
    pre_arr2 =np.ravel(pre_arr2)
    arr1,arr2=np.split(x_y_responses,2,axis=1)
    arr1 = np.ravel(arr1)
    arr2 = np.ravel(arr2)

    param_pre, _param_pre = scipy.optimize.curve_fit(gama_fit,pre_arr1,pre_arr2, bounds=(0,60))
    param, _param = scipy.optimize.curve_fit(gama_fit,arr1,arr2, bounds=(0,60))

    #pre_x = np.arange(int(np.floor(np.min(pre_arr1))),int(np.ceil(np.max(pre_arr2))),0.5)
    pre_x  = np.linspace(-0.5, 10,len(pre_arr2))
    #pre_x  = np.linspace(0, np.max(pre_arr2)+3,len(pre_arr2))
    #x = np.arange(int(np.floor(np.min(arr1))),int(np.ceil(np.max(arr1))),0.5)
    #x  = np.linspace(0, np.max(arr2)+3,len(arr2))
    x  = np.linspace(-0.5, 10,len(arr2))
    pre_y = gama_fit(pre_x,param_pre[0],param_pre[1],param_pre[2])
    y = gama_fit(x,param[0],param[1],param[2])
    if cell_type=="learners":
        color=bpf.CB_color_cycle[0]
    else:
        color= bpf.CB_color_cycle[1]
    axs.plot(pre_x, pre_y, color='k', linestyle='-', alpha=0.8, label="pre_training",linewidth=3)
    axs.plot(x, y, color=color, linestyle='-', alpha=0.8, label="post_training",linewidth=3)
    #axs[pat_num].text(1,10, f"r ={round(r_value*r_value,2)}", fontsize = 10) 
    return x, y

def plot_expected_vs_observed(pd_cell_data_mean_cell_grp,cell_type,f_norm_status,fig,axs1,axs2,axs3):
    pd_cell_data_mean_cell_grp = pd_cell_data_mean_cell_grp[pd_cell_data_mean_cell_grp["pre_post_status"]!="post_5"]
    pd_cell_data_mean_cell_grp = pd_cell_data_mean_cell_grp.groupby(by='cell_ID')
    axs=[axs1,axs2,axs3]
    
    all_resp_pat_0 =[]
    all_resp_pat_1 =[]
    all_resp_pat_2 =[]
    pre_all_resp_pat_0 =[]
    pre_all_resp_pat_1 =[]
    pre_all_resp_pat_2 =[]
    
    for c, cell in pd_cell_data_mean_cell_grp:
        pp_grps = cell.groupby(by="pre_post_status")
        cl = len(cell["pre_post_status"].unique())-2
        for pp, ppresp in pp_grps:
            if pp not in ['pre','post_3']:
                continue
            else:
                pp =pp
                #print(f"found pp stat")
                
            if pp=="pre":
                color=bpf.pre_color
            #elif pp=="post_4":
            #    cmx= int(pp.split("_")[-1])/cl
            #    color=colorFader(post_color,post_late,mix=cmx)
            else:
                #continue
                cmx= int(pp.split("_")[-1])/cl
                color=bpf.colorFader(bpf.post_color,bpf.post_late,mix=cmx)
                if cell_type=="learners":
                    color = bpf.CB_color_cycle[0]
                else:
                    color = bpf.CB_color_cycle[1]
            pats= ppresp[ppresp["frame_status"]=="pattern"]["frame_id"].unique()
            for pat in pats:
                pat_num = int(pat.split("_")[-1])
                point_list = bpf.map_points_to_patterns(pat)
                pat_val = float(ppresp[ppresp["frame_id"]==pat]["max_trace"])
                point_sum_val = float(np.sum(np.array((ppresp[ppresp["frame_id"].isin(point_list)]["max_trace"]))))
                pat_val_nrm = pat_val#-pat_val
                point_sum_val_nrm = point_sum_val#-pat_val
                #print(f"pat, point: {pat_val_nrm},{point_sum_val}")
                axs[pat_num].axline([0,0], [1,1], linestyle=':',
                                    color=bpf.CB_color_cycle[6], label="linear sum",linewidth=2)
                axs[pat_num].scatter(point_sum_val_nrm,pat_val_nrm,color=color, label=pp, alpha=0.8,linewidth=2)
                axs[pat_num].spines[['right', 'top']].set_visible(False)
                axs[pat_num].set_xlim(-1,12)
                axs[pat_num].set_ylim(-1,12)
                axs[pat_num].set_xticks(np.arange(-1,12,4))
                axs[pat_num].set_yticks(np.arange(-1,12,4))
                
                if pat_num==0:
                    axs[pat_num].set_ylabel("observed\nresponse (mV)", fontsize=14)
                    
                else:
                    axs[pat_num].set_yticklabels([])
                if cell_type=="learners":
                    axs[pat_num].set_title(None)
                    axs[pat_num].set_xlabel(None)
                    axs[pat_num].set_xticklabels([])
                else:
                    if pat_num==1:
                        axs[pat_num].set_xlabel("expected response (mV)")
                    else:
                        axs[pat_num].set_title(None)
                if pp!="pre":
                    if pat_num==0:
                        all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
                    elif pat_num==1:
                        all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
                    elif pat_num==2:
                        all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
                    else:
                        continue
                elif pp=="pre":
                    if pat_num==0:
                        pre_all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
                    elif pat_num==1:
                        pre_all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
                    elif pat_num==2:
                        pre_all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
                    else:
                        continue
                    

    #poly_fit_sums(pre_all_resp_pat_0,all_resp_pat_0,0,fig, axs)
    #poly_fit_sums(pre_all_resp_pat_1,all_resp_pat_1,1,fig, axs)
    #poly_fit_sums(pre_all_resp_pat_2,all_resp_pat_2,2,fig, axs)
    eq_pre_max = np.max(pre_all_resp_pat_0)
    eq_post_max = np.max(all_resp_pat_0)
    eq_fit(pre_all_resp_pat_0,all_resp_pat_0,0,eq_pre_max,eq_post_max,cell_type, fig, axs1)
    eq_fit(pre_all_resp_pat_1,all_resp_pat_1,1,eq_pre_max,eq_post_max,cell_type, fig, axs2)
    eq_fit(pre_all_resp_pat_2,all_resp_pat_2,2,eq_pre_max,eq_post_max,cell_type, fig, axs3)

    
def plot_initial_final_wt(feature_extracted_data,sc_data_dict,fig,axs):
    learners=sc_data_dict["ap_cells"]["cell_ID"].unique()
    non_learners=sc_data_dict["an_cells"]["cell_ID"].unique()
    deselect_list = ["no_frame","inR"]
    feature_extracted_data=feature_extracted_data[~feature_extracted_data["frame_status"].isin(deselect_list)]
    cell_grp = feature_extracted_data.groupby(by="cell_ID")
    for cell, cell_data in cell_grp:
        if cell in learners:
            color=bpf.CB_color_cycle[0]
        elif cell in non_learners:
            color=bpf.CB_color_cycle[1]
        else:
            continue
        frame_grp = cell_data.groupby(by="frame_status")
        for frame, frame_data in frame_grp:
            if "point" not in frame:
                continue
            else:
                point_num = int(frame.split("_")[-1])
                pps_grp=frame_data.groupby(by="pre_post_status")
                for pps,pps_data in pps_grp:
                    axs.scatter(point_num,pps_data["max_trace"])



def plot_figure_7(extracted_feature_pickle_file_path,
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
                    ]
                     # Adjust these values as needed
    
    width_ratios = [1, 1, 1, 1, 1, 
                    1, 1, 1
                   ]# Adjust these values as needed

    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(10, 8,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.1, hspace=1)



    #plot patterns
    axs_pat_1 = fig.add_subplot(gs[0:1,0:1])
    axs_pat_2 = fig.add_subplot(gs[0:1,2:3])
    axs_pat_3 = fig.add_subplot(gs[0:1,4:5])

    plot_patterns(axs_pat_1,axs_pat_2,axs_pat_3,0.05,0.03,1)



    #plot summation for learners and leaners
    axs_ex_sm1 = fig.add_subplot(gs[1:3,0:2])
    axs_ex_sm2 = fig.add_subplot(gs[1:3,2:4])
    axs_ex_sm3 = fig.add_subplot(gs[1:3,4:6])
    plot_expected_vs_observed(sc_data_dict["ap_cells"],"learners","no_fnorm",
                              fig,axs_ex_sm1,axs_ex_sm2,axs_ex_sm3)
    axs_ex_sm_l_list = [axs_ex_sm1,axs_ex_sm2,axs_ex_sm3]
    label_axis(axs_ex_sm_l_list, "A")
    #axs_ex_sm2.set_title("learners")
    axs_ex_sm2.set_xlabel(None)
    axs_ex_sm4 = fig.add_subplot(gs[3:5,0:2])
    axs_ex_sm5 = fig.add_subplot(gs[3:5,2:4])
    axs_ex_sm6 = fig.add_subplot(gs[3:5,4:6])
    plot_expected_vs_observed(sc_data_dict["an_cells"],"non-learners","no_fnorm",
                              fig,axs_ex_sm4,axs_ex_sm5,axs_ex_sm6)
    #axs_ex_sm5.set_title("non-learners")
    
    axs_ex_sm_nl_list= [axs_ex_sm4,axs_ex_sm5,axs_ex_sm6]
    label_axis(axs_ex_sm_nl_list,"B")



    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_7.png"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser.
    description = '''Generates figure 7'''
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
    globoutdir= globoutdir/'Figure_7'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_7(pklpath,scpath,cell_stat_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
