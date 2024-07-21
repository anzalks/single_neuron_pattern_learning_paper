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
import baisic_plot_fuctnions_and_features as bpf

# plot features are defines in bpf
bpf.set_plot_properties()

vlinec = "#C35817"

cell_to_plot = "2023_02_24_cell_2" 

time_to_plot = 0.15 # in s

class Args: pass
args_ = Args()

def plot_image(image,axs_img,xoffset,yoffset,pltscale):
    axs_img.imshow(image, cmap='gray')
    pos = axs_img.get_position()  # Get the original position
    new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
               pos.height*pltscale]
    # Shrink the plot
    axs_img.set_position(new_pos)
    axs_img.axis('off')

def label_axis(axis_list,letter_label):
    for axs_no, axs in enumerate(axis_list):
        axs_no = axs_no+1
        axs.text(0.1,1,f'{letter_label}{axs_no}',transform=axs.transAxes,    
                      fontsize=16, fontweight='bold', ha='center', va='center')

def plot_patterns(axs_pat1,axs_pat2,axs_pat3,xoffset,yoffset):
    pattern_list = ["trained pattern","Overlapping pattern",
                    "Non-overlapping pattern"]
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
            trace = bpf.substract_baseline(trace,bl_period_in_ms=2)
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

def inset_plot_traces(cell_data,field_to_plot,ylim,xlim,
                      axs, pat_num,xoffset,yoffset,pltscale):
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
                trace = bpf.substract_baseline(trace,bl_period_in_ms=2)
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




def plot_figure_1(pickle_file_path,image_file_path,
                  projection_image,
                  outdir,cell_to_plot=cell_to_plot):
    cell_data = pd.read_pickle(pickle_file_path)
    deselect_lsit = ["no_frame","inR"]
    cell_data = cell_data[(cell_data["cell_ID"]==cell_to_plot)&(~cell_data["frame_id"].isin(deselect_lsit))]
    cell_data.reset_index()
    sampling_rate = int(cell_data["sampling_rate(Hz)"].unique())
    image= pillow.Image.open(image_file_path)
    proj_img = pillow.Image.open(projection_image).convert('L')
    # Define the width and height ratios
    width_ratios = [4, 4, 2, 2, 2, 
                    2, 2, 2, 2]  # Adjust these values as needed
    height_ratios = [1, 1, 1,1, 1
                    ]       # Adjust these values as needed

    fig = plt.figure(figsize=(14,6))
    gs = GridSpec(5, 9,width_ratios=width_ratios, 
                  height_ratios=height_ratios,figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    axs_img = fig.add_subplot(gs[0:2, 0:2])
    plot_image(image,axs_img,-0.01,-0.175,1.01)
    axs_img.text(0.05,1.1,'A',transform=axs_img.transAxes,    
             fontsize=16, fontweight='bold', ha='center', va='center')
   
    axs_proj = fig.add_subplot(gs[2:4,0:2])
    plot_image(proj_img,axs_proj,0.02, -0.175,0.75)
    axs_proj.text(0.05,1.1,'B',transform=axs_proj.transAxes,    
             fontsize=16, fontweight='bold', ha='center', va='center')

    
    axs_pat1=fig.add_subplot(gs[0:1,2:3])
    axs_pat2=fig.add_subplot(gs[0:1,4:5])
    axs_pat3=fig.add_subplot(gs[0:1,6:7])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0.05,0)
    
    axs_fl1=fig.add_subplot(gs[1:3,2:4])
    axs_fl2=fig.add_subplot(gs[1:3,4:6])
    axs_fl3=fig.add_subplot(gs[1:3,6:8])
    ylim = (-0.7,0.3) # in mV
    xlim = 150 # in mseconds
    ylabel="field response (mV)"
    plot_trace_raw_all_pats(cell_data,"field_trace(mV)", ylim, xlim,
                            ylabel,axs_fl1,axs_fl2,axs_fl3)
    axs_fl1.set_xlabel(None)
    axs_fl2.set_xlabel(None)
    axs_fl3.set_xlabel(None)
    axs_fl1.set_xticklabels([])
    axs_fl2.set_xticklabels([])
    axs_fl3.set_xticklabels([])
    axs_fl2.set_ylabel(None)
    axs_fl3.set_ylabel(None)
    axs_fl2.set_yticklabels([])
    axs_fl3.set_yticklabels([])
    axs_fl_list = [axs_fl1,axs_fl2,axs_fl3]
    label_axis(axs_fl_list,"C")

    axs_inset = fig.add_subplot(gs[1:3,2:4])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",ylim,xlim, 
                      axs_inset,0,0.08,0.08,0.3)

    axs_inset = fig.add_subplot(gs[1:3,4:6])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",ylim,xlim, 
                      axs_inset,1,0.08,0.08,0.3)
    axs_inset = fig.add_subplot(gs[1:3,6:8])
    ylim = (-0.7,0.3)
    xlim = 15
    inset_plot_traces(cell_data,"field_trace(mV)",ylim,xlim, 
                      axs_inset,2,0.08,0.08,0.3)



    axs_cl1=fig.add_subplot(gs[3:5,2:4])
    axs_cl2=fig.add_subplot(gs[3:5,4:6])
    axs_cl3=fig.add_subplot(gs[3:5,6:8])
    ylim = (-2,4) # in mV
    xlim = 150 # in mseconds
    ylabel="cell response (mV)"
    plot_trace_raw_all_pats(cell_data,"cell_trace(mV)", ylim, xlim,
                            ylabel,axs_cl1,axs_cl2,axs_cl3)
    axs_cl1.set_xlabel(None)
    axs_cl3.set_xlabel(None)
    axs_cl2.set_ylabel(None)
    axs_cl3.set_ylabel(None)
    axs_cl2.set_yticklabels([])
    axs_cl3.set_yticklabels([])
    axs_cl_list = [axs_cl1,axs_cl2,axs_cl3]
    label_axis(axs_cl_list,"D")


    handles, labels = axs_cl2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs_cl2.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor =(0.8, 1),
               ncol = 1,title="Voltage trace",
               loc='upper center',frameon=False)#,loc='lower center'    

    plt.tight_layout()
    outpath = f"{outdir}/figure_1.png"
    outpath = f"{outdir}/figure_1.svg"
    outpath = f"{outdir}/figure_1.pdf"
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
                        , help = 'path to the giant pickle file with all cells'
                       )
    parser.add_argument('--illustration-path', '-i'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file in png format'
                       )
    parser.add_argument('--projection-image', '-p'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file showing projections'
                       )


    parser.add_argument('--outdir-path','-o'
                        ,required = False, default ='./', type=str
                        ,help = 'where to save the generated figure image'
                       )
    #    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    pklpath = Path(args.pikl_path)
    illustration_path = Path(args.illustration_path)
    projection_path = Path(args.projection_image)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_1'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_1(pklpath,illustration_path,projection_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
