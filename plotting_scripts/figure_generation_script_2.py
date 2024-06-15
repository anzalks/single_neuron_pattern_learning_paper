__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Generates the figure 2 of pattern learning paper.
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

time_points = ["pre","0", "10", "20","30" ]
selected_time_points = ['post_0', 'post_1', 'post_2', 'post_3','pre']
                        #'post_4','post_5']

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
        axs.text(-0.1,1.05,f'{letter_label}{axs_no}',transform=axs.transAxes,    
                      fontsize=16, fontweight='bold', ha='center', va='center')

def plot_raw_trace_time_points(single_cell_df,
                               deselect_list,fig,gs):
    single_cell_df = single_cell_df[~single_cell_df["frame_status"].isin(deselect_list)]
    sampling_rate = 20000 # for patterns
    sc_pat_grp = single_cell_df.groupby(by="frame_id")
    for pat, pat_data in sc_pat_grp:
        if "pattern" in pat:
            pat_num = int(pat.split('_')[-1])
        else:
            continue
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
                axs_trace.plot(time,pre_trace, color=bpf.pre_color,
                               label="baseline response")
                if pat_num==1:
                    axs_trace.set_ylabel("membrane potential(mV)")
                else:
                    axs_trace.set_ylabel(None)
                if pat_num ==0:
                    axs_trace.set_title("pre")
                    axs_trace.text(-2,1.4,'B',transform=axs_trace.transAxes,    
                                fontsize=16, fontweight='bold', ha='center', va='center')            
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
                              alpha=0.6,label="baseline response")
                axs_trace.plot(time,trace,
                               color=bpf.post_late,
                               label="post training response")
                               #color=bpf.colorFader(bpf.post_color,
                               #                     bpf.post_late,
                               #                     (idx/len(pps_grp))))
                axs_trace.set_ylabel(None)
                axs_trace.set_yticklabels([])
                if pat_num==0:
                    if idx==1:
                        axs_trace.set_title(f"time points\n{time_points[idx+1]}")
                    else:
                        axs_trace.set_title(time_points[idx+1])
                else:
                    axs_trace.set_title(None)
            if (pat_num==2)and(idx==1):
                axs_trace.set_xlabel("time (ms)")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axs_trace.legend(by_label.values(), by_label.keys(), 
                           bbox_to_anchor =(0.1, -1),
                           ncol = 6,#title="cell response",
                           loc='upper center',frameon=False)
            elif pat_num ==2:
                axs_trace.set_xlabel(None)
            else:
                axs_trace.set_xlabel(None)
                axs_trace.set_xticklabels([])
            axs_trace.set_ylim(-2,6)
            axs_trace.spines[['right', 'top']].set_visible(False)







#plot_order = df.sort_values(by='Amount', ascending=False).ID.values
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
                 
def plot_cell_type_features(cell_list,pattern_number, fig, axs_slp,val_to_plot,plt_color):
    if pattern_number == "pattern_0":
        pat_type = "trained"
    elif pattern_number == "pattern_1":
        pat_type = "overlapping"
    else:
        pat_type = "untrained"
    #y_lim=(-50,300)
    y_lim = (-50,500)
    pat_num=int(pattern_number.split("_")[-1])
    num_cells= len(cell_list["cell_ID"].unique())
    #cell_type =get_variable_name(cell_list)
    pfd = cell_list.groupby(by="frame_id")
    for c, pat in pfd:
        if c != pattern_number:
            continue

        else:
            #pat = pat[(pat["pre_post_status"]!="post_5")]#&(pat["pre_post_status"]!="post_4")]#&(cell["pre_post_status"]!="post_3")]
            #order = np.array(('pre','post_0','post_1','post_2','post_3','post_4'),dtype=object)
            order = np.array(('pre','post_0','post_1','post_2','post_3'),dtype=object)
            #print(f"pat = &&&&&&&{pat}%%%%%%%%%%%%%")
            g=sns.stripplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
                            order=order,ax=axs_slp,color=bpf.CB_color_cycle[2],
                            alpha=0.6,size=5, label='single cell')#alpha=0.8,
            sns.pointplot(data=pat, x="pre_post_status",y=f"{val_to_plot}",
                          errorbar="se",order=order,capsize=0.1,ax=axs_slp,
                          color=plt_color, linestyles='dotted',scale = 0.8,
                         label="average of\nall cells")
            #palette="pastel",hue="cell_ID")
            g.legend_.remove()
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
            annotator = Annotator(axs_slp,anotp_list,data=pat, 
                                  x="pre_post_status",
                                  y=f"{val_to_plot}",
                                  order=order,
                                 fontsize=8)
            #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
            #annotator.annotate()

            #"""
            axs_slp.axhline(100, ls=':',color="k", alpha=0.4)
            if pat_num==0:
                sns.despine(fig=None, ax=axs_slp, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs_slp.set_ylabel("% change in\nEPSP amplitude")
                axs_slp.set_xlabel(None)
                #axs[pat_num].set_yticks([])
            elif pat_num==1:
                sns.despine(fig=None, ax=axs_slp, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs_slp.set_ylabel(None)
                axs_slp.set_xlabel("time points (mins)")
            elif pat_num==2:
                sns.despine(fig=None, ax=axs_slp, top=True, right=True, 
                            left=False, bottom=False, offset=None, trim=False)
                axs_slp.set_xlabel(None)
                axs_slp.set_ylabel(None)
                handles, labels = g.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axs_slp.legend(by_label.values(), by_label.keys(), 
                               bbox_to_anchor =(0.6, 1),
                               ncol = 1,title="cell response",
                               loc='upper center',frameon=False)


            else:
                #g.legend_.remove()
                pass 
            g.set(ylim=y_lim)
            g.set_xticklabels(time_points,rotation=30)
    #handles, labels = g.get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #axs_slp.legend(by_label.values(), by_label.keys(), 
    #               bbox_to_anchor =(0.1, 0.8),
    #               ncol = 6,title="cell response",
    #               loc='upper center',frameon=False)
            #g.legend_.remove()
            
        
            
def plot_field_normalised_feature_multi_patterns(cell_list,val_to_plot,
                                                fig,axs1,axs2,axs3):
    cell_list= norm_values(cell_list,val_to_plot)
    plot_cell_type_features(cell_list,"pattern_0",fig, axs1,val_to_plot,
                            bpf.CB_color_cycle[5])
    plot_cell_type_features(cell_list,"pattern_1",fig, axs2,val_to_plot,
                            bpf.CB_color_cycle[5])
    plot_cell_type_features(cell_list,"pattern_2",fig, axs3,val_to_plot,
                            bpf.CB_color_cycle[5])
    

#["cell_ID","frame_status","pre_post_status","frame_id","min_trace","max_trace","abs_area","pos_area",
#"neg_area","onset_time","max_field","min_field","slope","intercept","min_trace_t","max_trace_t","max_field_t","min_field_t","mean_trace","mean_field","mean_ttl","mean_rmp"]

def inR_sag_plot(inR_all_Cells_df,fig,axs):
    deselect_list = ['post_4','post_5']
    inR_all_Cells_df =inR_all_Cells_df[~inR_all_Cells_df["pre_post_status"].isin(deselect_list)] 
    order = np.array(('pre','post_0', 'post_1', 'post_2', 'post_3'),dtype=object)

    g=sns.pointplot(data=inR_all_Cells_df,x="pre_post_status",y="inR",
                    capsize=0.2,errorbar=('sd'),order=order,color="k",
                    label="input\nresistance")
    sns.pointplot(data=inR_all_Cells_df,x="pre_post_status",y="sag",
                  capsize=0.2,errorbar=('sd'),order=order,
                  color=bpf.CB_color_cycle[4], label="sag value")
    sns.stripplot(data=inR_all_Cells_df,color=bpf.CB_color_cycle[4],
                  x="pre_post_status",y="sag",
                  order=order,alpha=0.2)

    pre_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"]=="pre"]["sag"]
    post_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"]=="post_3"]["sag"]
    pre= spst.wilcoxon(pre_trace,post_trace,zero_method="wilcox", correction=True)
    pvalList=pre.pvalue
    print(pvalList)
    anotp_list=("pre","post_3")
    annotator = Annotator(axs,[anotp_list],data=inR_all_Cells_df, x="pre_post_status",y="sag",order=order)
    #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()
    axs.legend_.remove()

    #sns.move_legend(axs, "upper left", bbox_to_anchor=(1, 1))
    #axs.set_ylim(-10,250)
    axs.set_xticklabels(time_points)
    sns.despine(fig=None, ax=axs, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    axs.set_ylabel("MOhms")
    axs.set_xlabel("time points (mins)")
    handles, labels = g.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys(), 
                   bbox_to_anchor =(0.5, 1.275),
                   ncol = 2,
                   loc='upper center',frameon=False)




    inr_pos = axs.get_position()
    new_inr_pos = [inr_pos.x0, inr_pos.y0-0.04, inr_pos.width,
                   inr_pos.height]
    axs.set_position(new_inr_pos)
    
    

def plot_figure_2(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  inR_all_Cells_df,
                  illustration_path,
                  inRillustration_path,
                  outdir,cell_to_plot=cell_to_plot):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    single_cell_df = feature_extracted_data.copy()
    single_cell_df = single_cell_df[(single_cell_df["cell_ID"]==cell_to_plot)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    sc_data = pd.read_pickle(cell_categorised_pickle_file)
    sc_data_df = pd.concat([sc_data["ap_cells"],
                            sc_data["an_cells"]]).reset_index(drop=True)
    inR_all_Cells_df = pd.read_pickle(inR_all_Cells_df) 
    illustration = pillow.Image.open(illustration_path)
    inRillustration = pillow.Image.open(inRillustration_path)
    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 1, 0.8]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 
                     0.5, 0.5, 0.5, 0.5, 0.2]       # Adjust these values as needed

    fig = plt.figure(figsize=(8,18))
    gs = GridSpec(11, 7,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.2)
    #place illustration
    axs_img = fig.add_subplot(gs[:3, :6])
    plot_image(illustration,axs_img,0,-0.01,1)

    axs_img.text(-0.07,0.95,'A',transform=axs_img.transAxes,    
            fontsize=16, fontweight='bold', ha='center', va='center')

    axs_vpat1=fig.add_subplot(gs[3,0])
    axs_vpat2=fig.add_subplot(gs[4,0])
    axs_vpat3=fig.add_subplot(gs[5,0])
    plot_patterns(axs_vpat1,axs_vpat2,axs_vpat3,-0.075,0,2)
    
    plot_raw_trace_time_points(single_cell_df,deselect_list,fig,gs)
    #plot pattern projections 
    axs_pat1 = fig.add_subplot(gs[6,0])
    axs_pat2 = fig.add_subplot(gs[6,2])
    axs_pat3 = fig.add_subplot(gs[6,4])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0.05,-0.03,1)
    
    

    #plot amplitudes over time
    feature_extracted_data =feature_extracted_data[~feature_extracted_data["frame_status"].isin(deselect_list)]
    cell_grp = feature_extracted_data.groupby(by="cell_ID")
    axs_slp1 = fig.add_subplot(gs[7:9,0:2])
    axs_slp1.set_ylabel("slope (mV/ms)")
    axs_slp2 = fig.add_subplot(gs[7:9,2:4])
    axs_slp2.set_yticklabels([])
    axs_slp3 = fig.add_subplot(gs[7:9,4:6])
    axs_slp3.set_yticklabels([])
    plot_field_normalised_feature_multi_patterns(sc_data_df,"max_trace",
                                                 fig,axs_slp1,axs_slp2,
                                                 axs_slp3)
    axs_slp_list = [axs_slp1,axs_slp2,axs_slp3]
    label_axis(axs_slp_list,"C")


    axs_inr = fig.add_subplot(gs[9:10,3:6])
    inR_sag_plot(inR_all_Cells_df,fig,axs_inr)
    axs_inr.text(-0.05,1.1,'E',transform=axs_inr.transAxes,    
             fontsize=16, fontweight='bold', ha='center', va='center')            


    axs_inrill = fig.add_subplot(gs[9:10,0:3])
    plot_image(inRillustration,axs_inrill,-0.05,-0.05,1)
    axs_inrill.text(-0.05,1.1,'D',transform=axs_inrill.transAxes,    
                 fontsize=16, fontweight='bold', ha='center', va='center')            


    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
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
    parser.add_argument('--sortedcell-path', '-s'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with cell sorted'
                        'exrracted data'
                       )
    parser.add_argument('--inR-path', '-r'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with inR data'
                       )


    parser.add_argument('--illustration-path', '-i'
                        , required = False,default ='./', type=str
                        , help = 'path to the image file in png format'
                       )

    parser.add_argument('--inRillustration-path', '-p'
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
    inR_path = Path(args.inR_path)
    illustration_path = Path(args.illustration_path)
    inRillustration_path = Path(args.inRillustration_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_2'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_2(pklpath,scpath,inR_path,illustration_path,inRillustration_path,globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
