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

cell_to_plot = "2022_12_21_cell_1" 

time_to_plot = 0.15 # in s 

time_points = ["pre","0 mins", "10 mins", "20 mins","30 mins" ]
selected_time_points = ['post_0', 'post_1', 'post_2', 'post_3','pre']
                        #'post_4','post_5']

class Args: pass
args_ = Args()


def plot_figure_2(extracted_feature_pickle_file_path,
                  illustration_path,
                  outdir,cell_to_plot=cell_to_plot):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    single_cell_df = feature_extracted_data.copy()
    single_cell_df = single_cell_df[(single_cell_df["cell_ID"]==cell_to_plot)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    illustration = pillow.Image.open(illustration_path).convert('L')
    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 1, 1]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5]       # Adjust these values as needed

    fig = plt.figure(figsize=(8,18))
    gs = GridSpec(10, 7,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.3)
    #place illustration
    ax_img = fig.add_subplot(gs[:3, :6])
    ax_img.imshow(illustration, cmap='gray')

    ax_img.axis('off')
    ax_img.text(0.05,1.15,'A',transform=ax_img.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')


    #plot trace at different time points
    single_cell_df = single_cell_df[~single_cell_df["frame_status"].isin(deselect_list)]
    sampling_rate = 20000 # for patterns
    sc_pat_grp = single_cell_df.groupby(by="frame_id")
    for pat, pat_data in sc_pat_grp:
        pat_num = int(pat.split('_')[-1])
        pre_trace  =pat_data[pat_data["pre_post_status"]=="pre"]["mean_trace"][0]
        print(f"pre_trace = {pre_trace}")
        pps_grp = pat_data.groupby(by="pre_post_status")
        for idx, pps_data in enumerate(pps_grp):
            if pps_data[0]=="pre":
                axs_trace = fig.add_subplot(gs[3+pat_num,1])
                trace = pps_data[-1]["mean_trace"][0]
                trace = bpf.substract_baseline(trace)
                trace = trace[:int(sampling_rate*time_to_plot)]
                pre_trace = bpf.substract_baseline(pre_trace)
                pre_trace = pre_trace[:int(sampling_rate*time_to_plot)]
                time = np.linspace(0,time_to_plot,len(trace))*1000
                axs_trace.plot(time,pre_trace, color=bpf.pre_color)
                if pat_num==1:
                    axs_trace.set_ylabel("membrane potential(mV)")
                else:
                    axs_trace.set_ylabel(None)
                if pat_num ==0:
                    axs_trace.set_title("pre")
                else:
                    axs_trace.set_title(None)
            else:
                axs_trace = fig.add_subplot(gs[3+pat_num,idx+2])
                trace = pps_data[-1]["mean_trace"][0]
                trace = bpf.substract_baseline(trace)
                trace = trace[:int(sampling_rate*time_to_plot)]
                pre_trace = bpf.substract_baseline(pre_trace)
                pre_trace = pre_trace[:int(sampling_rate*time_to_plot)]
                time = np.linspace(0,time_to_plot,len(trace))*1000
                axs_trace.plot(time,pre_trace, color=bpf.pre_color,
                              alpha=0.6)
                axs_trace.plot(time,trace,
                               color=bpf.post_late)
                               #color=bpf.colorFader(bpf.post_color,
                               #                     bpf.post_late,
                               #                     (idx/len(pps_grp))))
                axs_trace.set_ylabel(None)
                axs_trace.set_yticklabels([])
                if pat_num==0:
                    axs_trace.set_title(time_points[idx+1])
                else:
                    axs_trace.set_title(None)
            if (pat_num==2)and(idx==1):
                axs_trace.set_xlabel("time (ms)")
            elif pat_num ==2:
                axs_trace.set_xlabel(None)
            else:
                axs_trace.set_xlabel(None)
                axs_trace.set_xticklabels([])
            axs_trace.set_ylim(-2,6)
            axs_trace.spines[['right', 'top']].set_visible(False)
        
    pattern_list = ["trained\npattern","Overlapping\npattern",
                    "Non-overlapping\npattern"]
    for pr_no, pattern in enumerate(pattern_list):
        if pr_no==0:
            axs_pat = fig.add_subplot(gs[pr_no+3,0])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(0,2)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = fig.add_subplot(gs[pr_no+3,0])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,2)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = fig.add_subplot(gs[pr_no+3,0])  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,2)
            axs_pat.imshow(pat_fr)
        else:
            print("exception in pattern number")
        pat_pos = axs_pat.get_position()
        new_pat_pos = [pat_pos.x0-0.07, pat_pos.y0, pat_pos.width,
                        pat_pos.height]
        axs_pat.set_position(new_pat_pos)
        axs_pat.axis('off')
        #axs_pat.set_title(pattern)
    
    #plot pattern projections 
    pattern_list = ["trained\npattern","Overlapping\npattern",
                    "Non-overlapping\npattern"]
    for p_no, pattern in enumerate(pattern_list):
        if p_no==0:
            axs_proj = fig.add_subplot(gs[6,0])  #plt.subplot2grid((3,4),(0,p_no))
            proj = bpf.create_grid_image(0,2)
        elif p_no==1:
            axs_proj = fig.add_subplot(gs[6,2])  #plt.subplot2grid((3,4),(0,p_no))
            proj = bpf.create_grid_image(4,2)
        else:
            axs_proj = fig.add_subplot(gs[6,4])  #plt.subplot2grid((3,4),(0,p_no))
            proj = bpf.create_grid_image(17,2)
        axs_proj.imshow(proj)
        proj_pos = axs_proj.get_position()
        new_proj_pos = [proj_pos.x0+0.05, proj_pos.y0, proj_pos.width,
                        proj_pos.height]
        axs_proj.set_position(new_proj_pos)
        axs_proj.axis('off')
        axs_proj.set_title(pattern)

    #plot amplitudes over time
    feature_extracted_data =feature_extracted_data[~feature_extracted_data["frame_status"].isin(deselect_list)]
    cell_grp = feature_extracted_data.groupby(by="cell_ID")
    axs_slp1 = fig.add_subplot(gs[7:9,0:2])
    axs_slp2 = fig.add_subplot(gs[7:9,2:4])
    axs_slp3 = fig.add_subplot(gs[7:9,4:6])
    










    #
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.215, 0.2),
    #           ncol = 2,title="Voltage trace",
    #           loc='upper center',frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_2.png"
    plt.savefig(outpath,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def main():
    # Argument parser.
    description = '''Generates figure 1'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pikl-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with extracted features'
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
    illustration_path = Path(args.illustration_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_2'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_2(pklpath,illustration_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
