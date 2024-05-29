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
    width_ratios = [1, 4, 4, 4, 4]  # Adjust these values as needed
    height_ratios = [1, 1, 1]       # Adjust these values as needed

    fig = plt.figure(figsize=(14,6))
    gs = GridSpec(3, 5,width_ratios=width_ratios, 
                  height_ratios=height_ratios,figure=fig)
    gs.update(wspace=0.2, hspace=0.8)

    ax_img = fig.add_subplot(gs[:2, :2])
    
    ax_img.imshow(image, cmap='gray')
    pos = ax_img.get_position()  # Get the original position
    new_pos = [pos.x0, pos.y0+0.1, pos.width * 0.8, pos.height * 0.8]
    # Shrink the plot
    ax_img.set_position(new_pos)

    ax_img.axis('off')
    ax_img.text(0.05,1.1,'A',transform=ax_img.transAxes,    
             fontsize=16, fontweight='bold', ha='center', va='center')
   
    ax_img1 = fig.add_subplot(gs[1,1])
    
    ax_img1.imshow(proj_img, cmap='gray')
    pos1 = ax_img1.get_position()  # Get the original position
    new_pos1 = [pos.x0, pos.y0-0.275, pos.width * 0.8, pos.height * 0.8]
    # Shrink the plot
    ax_img1.set_position(new_pos1)
    ax_img1.axis('off')
    ax_img1.text(0.05,-0.15,'B',transform=ax_img.transAxes,    
             fontsize=16, fontweight='bold', ha='center', va='center')


    pattern_list = ["trained pattern","Overlapping pattern",
                    "Non-overlapping pattern"]
    for p_no, pattern in enumerate(pattern_list):
        axs = fig.add_subplot(gs[0,p_no+2])  #plt.subplot2grid((3,4),(0,p_no))
        if p_no==0:
            img = bpf.create_grid_image(0,2)
        elif p_no==1:
            img = bpf.create_grid_image(4,2)
        else:
            img = bpf.create_grid_image(17,2)
        axs.imshow(img)
        axs.axis('off')
        axs.set_title(pattern)
    
    pattern_grps = cell_data.groupby(by="frame_id")
    for pat,pat_data in pattern_grps:
        axs_no = int(pat.split('_')[-1])+2
        axs = fig.add_subplot(gs[1,axs_no])  #plt.subplot2grid((3,4),(1,axs_no))
        trial_grp = pat_data.groupby(by="trial_no")
        mean_trace = []
        for tr, trial_data in trial_grp:
            trace = trial_data["field_trace(mV)"].to_numpy()[:int(time_to_plot*sampling_rate)]
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
        if int(pat.split('_')[-1])==0:
            axs.set_ylabel("field response (mV)")
        else:
            axs.set_xlabel(None)
            axs.set_yticklabels([])
        axs.set_ylim(-0.5,0.3)
        axs.text(0,1.3,chr(67 + (axs_no - 2)),transform=axs.transAxes,
                fontsize=16, fontweight='bold', ha='center', va='center')


    for pat,pat_data in pattern_grps:
        axs_no = int(pat.split('_')[-1])+2
        axs = fig.add_subplot(gs[2,axs_no])  #plt.subplot2grid((3,4),(2,axs_no))
        trial_grp = pat_data.groupby(by="trial_no")
        mean_trace = []
        for tr, trial_data in trial_grp:
            trace = trial_data["cell_trace(mV)"].to_numpy()[:int(time_to_plot*sampling_rate)]
            trace = bpf.substract_baseline(trace)
            mean_trace.append(trace)
            time = np.linspace(0,time_to_plot,len(trace))*1000
            axs.plot(time, trace, color='k',alpha=0.2,label='trials')
        mean_trace = np.array(mean_trace)
        mean_trace = np.mean(mean_trace,axis=0)
        time = np.linspace(0,time_to_plot,len(mean_trace))*1000
        axs.plot(time,mean_trace,color='k',label='mean')
        axs.axvline(0,color=vlinec,linestyle=':',label ='End of light\npulse')
        axs.spines[['right', 'top']].set_visible(False)
        if int(pat.split('_')[-1])==0:
            axs.set_ylabel("membrane \n voltage (mV)")
        else:
            axs.set_xlabel(None)
            axs.set_yticklabels([])
        axs.set_ylim(-1,4)
        axs.set_xlabel("time(ms)")
        axs.text(0,1.3,chr(70 + (axs_no - 2)),transform=axs.transAxes,
                fontsize=16, fontweight='bold', ha='center', va='center')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               bbox_to_anchor =(0.2, 0.2),
               ncol = 2,title="Voltage trace",
               loc='upper center')#,loc='lower center'    
    

    plt.tight_layout()
    outpath = f"{outdir}/figure_1.png"
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
