__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

"""
Figure 2: EPSP Amplitude Analysis

This script generates Figure 2 of the pattern learning paper, which shows:
- EPSP amplitude changes across different time points (pre, post-0, post-1, post-2, post-3)
- Pattern-specific responses for trained, overlapping, and non-overlapping patterns
- Cell type classification and feature analysis
- Input resistance measurements and their correlation with learning
- Statistical comparisons between learners and non-learners

Input files:
- pd_all_cells_mean.pickle: Mean responses for all cells
- all_cells_classified_dict.pickle: Cell classification (learners/non-learners)
- all_cells_inR.pickle: Input resistance data
- Illustration images: Pattern and recording setup diagrams

Output: Figure_2/figure_2.png showing EPSP amplitude analysis and cell classification
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
from shared_utils import baisic_plot_fuctnions_and_features as bpf
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import levene
from PIL import ImageDraw, ImageFont
# Additional imports for new statistical tests
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

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


def perform_repeated_measures_anova(data, val_to_plot):
    """
    Perform repeated measures ANOVA and return p-values for pairwise comparisons
    """
    try:
        # Reshape data for repeated measures ANOVA
        # Create a long format dataframe
        long_data = []
        for cell_id in data['cell_ID'].unique():
            cell_data = data[data['cell_ID'] == cell_id]
            for _, row in cell_data.iterrows():
                long_data.append({
                    'subject': cell_id,
                    'time_point': row['pre_post_status'],
                    'value': row[val_to_plot]
                })
        
        long_df = pd.DataFrame(long_data)
        
        # Perform repeated measures ANOVA using pingouin
        aov = pg.rm_anova(dv='value', within='time_point', subject='subject', data=long_df)
        
        # If significant main effect, perform pairwise comparisons
        if aov['p-unc'].iloc[0] < 0.05:
            # Pairwise t-tests with Bonferroni correction
            posthoc = pg.pairwise_ttests(dv='value', within='time_point', subject='subject', 
                                       data=long_df, padjust='bonf')
            
            # Extract p-values for pre vs post comparisons
            pval_list = []
            time_points_order = ['pre', 'post_0', 'post_1', 'post_2', 'post_3']
            
            for time_point in time_points_order[1:]:  # Skip 'pre'
                comparison = posthoc[(posthoc['A'] == 'pre') & (posthoc['B'] == time_point)]
                if not comparison.empty:
                    pval_list.append(comparison['p-corr'].iloc[0])
                else:
                    # Try reverse comparison
                    comparison = posthoc[(posthoc['A'] == time_point) & (posthoc['B'] == 'pre')]
                    if not comparison.empty:
                        pval_list.append(comparison['p-corr'].iloc[0])
                    else:
                        pval_list.append(1.0)  # No significant difference
        else:
            # No significant main effect
            pval_list = [1.0] * 4  # Four comparisons (pre vs post_0, post_1, post_2, post_3)
            
        return pval_list
        
    except Exception as e:
        print(f"Error in repeated measures ANOVA: {e}")
        # Fall back to individual t-tests
        pval_list = []
        time_points_order = ['pre', 'post_0', 'post_1', 'post_2', 'post_3']
        
        for time_point in time_points_order[1:]:
            try:
                pre_data = data[data['pre_post_status'] == 'pre'][val_to_plot]
                post_data = data[data['pre_post_status'] == time_point][val_to_plot]
                _, pval = spst.ttest_rel(pre_data, post_data)
                pval_list.append(pval)
            except:
                pval_list.append(1.0)
        
        return pval_list


def perform_mixed_effect_model(data, val_to_plot):
    """
    Perform mixed effect model analysis and return p-values for pairwise comparisons
    Uses linear mixed model approach through statsmodels
    """
    try:
        # Prepare data for mixed effect model
        long_data = []
        for cell_id in data['cell_ID'].unique():
            cell_data = data[data['cell_ID'] == cell_id]
            for _, row in cell_data.iterrows():
                long_data.append({
                    'subject': cell_id,
                    'time_point': row['pre_post_status'],
                    'value': row[val_to_plot]
                })
        
        long_df = pd.DataFrame(long_data)
        
        # Use a simpler approach: linear mixed model through statsmodels
        from statsmodels.formula.api import mixedlm
        
        # Create dummy variables for time points
        long_df = pd.get_dummies(long_df, columns=['time_point'], prefix='time')
        
        # Build the model formula
        time_cols = [col for col in long_df.columns if col.startswith('time_')]
        
        # Reference category is 'pre' (excluded from model)
        formula_parts = []
        ref_col = None
        for col in time_cols:
            if 'pre' in col:
                ref_col = col
            else:
                formula_parts.append(col)
        
        if not formula_parts:
            raise ValueError("No time point columns found for mixed model")
        
        formula = f"value ~ {' + '.join(formula_parts)}"
        
        # Fit mixed model with random intercept for subject
        model = mixedlm(formula, long_df, groups=long_df['subject'])
        result = model.fit()
        
        # Extract p-values for each time point comparison vs reference (pre)
        pval_list = []
        time_points_order = ['post_0', 'post_1', 'post_2', 'post_3']
        
        for time_point in time_points_order:
            col_name = f'time_{time_point}'
            if col_name in result.params.index:
                pval = result.pvalues[col_name]
                pval_list.append(pval)
            else:
                pval_list.append(1.0)
        
        return pval_list
        
    except Exception as e:
        print(f"Error in mixed effect model: {e}")
        # Fall back to individual t-tests with FDR correction
        pval_list = []
        time_points_order = ['pre', 'post_0', 'post_1', 'post_2', 'post_3']
        
        raw_pvals = []
        for time_point in time_points_order[1:]:
            try:
                pre_data = data[data['pre_post_status'] == 'pre'][val_to_plot]
                post_data = data[data['pre_post_status'] == time_point][val_to_plot]
                _, pval = spst.ttest_rel(pre_data, post_data)
                raw_pvals.append(pval)
            except:
                raw_pvals.append(1.0)
        
        # Apply FDR correction
        from statsmodels.stats.multitest import fdrcorrection
        _, corrected_pvals = fdrcorrection(raw_pvals)
        
        return corrected_pvals.tolist()


def plot_cell_type_features_with_custom_stats(cell_list, pattern_number, fig, axs_slp, val_to_plot, plt_color, stat_method="wilcoxon"):
    """
    Modified version of plot_cell_type_features with different statistical test options
    stat_method: "wilcoxon", "rm_anova", or "mixed_effect"
    """
    if pattern_number == "pattern_0":
        pat_type = "trained"
    elif pattern_number == "pattern_1":
        pat_type = "overlapping"
    else:
        pat_type = "untrained"

    y_lim = (-50,700)
    time_points_= ["pre","0", "10", "20","30" ]
    pat_num = int(pattern_number.split("_")[-1])
    num_cells = len(cell_list["cell_ID"].unique())
    pfd = cell_list.groupby(by="frame_id")

    # Set a fixed y-axis limit for consistency across all plots
    axs_slp.set_ylim(y_lim)

    # Set a common base_y and step_y for all annotations
    base_y = 270  # Absolute y-axis position for the first annotation
    step_y = 40   # Spacing between each annotation

    # Define time points for x-tick labels
    time_point_list = ["pre", "post_0", "post_1", "post_2", "post_3"]

    for c, pat in pfd:
        if c != pattern_number:
            continue
        else:
            order = np.array(time_point_list, dtype=object)
            
            # Create the stripplot for individual cells (with legend label)
            stripplot = sns.stripplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}",
                order=order, ax=axs_slp, color=bpf.CB_color_cycle[2],
                alpha=0.6, size=5, label='single cell'
            )
            
            # Create the pointplot for the average of all cells (no automatic legend)
            sns.pointplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}",
                errorbar="se", order=order, capsize=0.1, ax=axs_slp,
                color=plt_color, linestyles='dotted', scale=0.8
            )
            
            # Create a custom marker for the point plot in the legend (instead of a line)
            pointplot_handle = Line2D([], [], marker='o', color=plt_color, linestyle='', label='average of\nall cells')

            # Set title, limits, and aesthetics
            stripplot.set_title(None)
            axs_slp.axhline(100, ls=':', color="k", alpha=0.4)
            stripplot.set(ylim=y_lim)

            # Set the custom x-ticks and labels based on time_points
            axs_slp.set_xticks(range(len(time_point_list)))  # Ensure correct positions for the labels
            axs_slp.set_xticklabels(time_points_, rotation=30)  # Map time_points as x-tick labels

            # Perform statistical tests based on selected method
            if stat_method == "wilcoxon":
                # Original Wilcoxon signed-rank test
                pvalList = []
                anotp_list = []
                for i in order[1:]:
                    posti = spst.wilcoxon(
                        pat[pat["pre_post_status"] == 'pre'][f"{val_to_plot}"],
                        pat[pat["pre_post_status"] == i][f"{val_to_plot}"],
                        zero_method="wilcox", correction=True
                    )
                    pvalList.append(posti.pvalue)
                    anotp_list.append(("pre", i))
            
            elif stat_method == "rm_anova":
                # Repeated measures ANOVA
                pvalList = perform_repeated_measures_anova(pat, val_to_plot)
                anotp_list = [("pre", i) for i in order[1:]]
            
            elif stat_method == "mixed_effect":
                # Mixed effect model
                pvalList = perform_mixed_effect_model(pat, val_to_plot)
                anotp_list = [("pre", i) for i in order[1:]]

            # Manually add annotations using matplotlib
            for idx, (pval, pair) in enumerate(zip(pvalList, anotp_list)):
                x1, x2 = pair
                x1_pos = order.tolist().index(x1)
                x2_pos = order.tolist().index(x2)

                # Draw the annotation line
                axs_slp.plot([x1_pos, x2_pos], [base_y + idx * step_y] * 2, color='black', linewidth=1)

                # Add the p-value text above the line
                annotation_text = bpf.convert_pvalue_to_asterisks(pval)
                axs_slp.text(
                    (x1_pos + x2_pos) / 2, base_y + idx * step_y + 2, 
                    annotation_text, ha='center', va='bottom', fontsize=8
                )

            # Adjust axis labels and despine
            if pat_num == 0:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_ylabel("% change in\nEPSP amplitude")
                axs_slp.set_xlabel(None)
                # Remove legend for panel Di
                if axs_slp.get_legend() is not None:
                    axs_slp.get_legend().remove()
            elif pat_num == 1:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_ylabel(None)
                axs_slp.set_yticklabels([])
                axs_slp.set_xlabel("time points (mins)")
                # Remove legend for panel Dii
                if axs_slp.get_legend() is not None:
                    axs_slp.get_legend().remove()
            elif pat_num == 2:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_xlabel(None)
                axs_slp.set_ylabel(None)

                # Get the handles and labels for the stripplot
                handles1, labels1 = stripplot.get_legend_handles_labels()

                # Remove duplicate "single cell" entries
                handles1_labels_dict = dict(zip(labels1, handles1))  # Ensure unique entries
                handles = list(handles1_labels_dict.values()) + [pointplot_handle]
                labels = list(handles1_labels_dict.keys()) + ['average of\nall cells']

                # Combine legends and add them to the plot
                axs_slp.legend(
                    handles, labels,
                    bbox_to_anchor=(0.8, 0.9),  # Adjust the legend position
                    ncol=1, title="cell response",
                    loc='upper center',
                    handletextpad=0.2,  # Reduce space between marker and text
                    labelspacing=0.2,  # Reduce vertical space between entries
                    fancybox=True,  # Enable fancy box with rounded corners
                    framealpha=0.7,  # Set the transparency of the legend background
                    facecolor='white'  # Set the background color to white
                )

    return axs_slp


def plot_field_normalised_feature_multi_patterns_with_stats(cell_list, val_to_plot, fig, axs1, axs2, axs3, stat_method="wilcoxon"):
    """
    Modified version that uses different statistical tests
    """
    cell_list = norm_values(cell_list, val_to_plot)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_0", fig, axs1, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_1", fig, axs2, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_2", fig, axs3, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)

#def plot_image(image,axs_img,xoffset,yoffset,pltscale):
#    axs_img.imshow(image, cmap='gray')
#    pos = axs_img.get_position()  # Get the original position
#    new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
#               pos.height*pltscale]
#    # Shrink the plot
#    axs_img.set_position(new_pos)
#    axs_img.axis('off')

def move_axis(axs_list,xoffset,yoffset,pltscale):
    for axs in axs_list:
        pos = axs.get_position()  # Get the original position
        new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
                   pos.height*pltscale]
        # Shrink the plot
        axs.set_position(new_pos)



def plot_image(image, axs_img, xoffset=0, yoffset=0, pltscale=1):
    """
    Plot a Pillow Image object on the provided axes, ensuring transparent areas are white.
    
    Parameters:
    - image (pillow.Image.Image): Pillow Image object.
    - axs_img (plt.Axes): Matplotlib axis object to plot the image.
    - xoffset (float): Horizontal offset for repositioning the image.
    - yoffset (float): Vertical offset for repositioning the image.
    - pltscale (float): Scaling factor for the image plot size.
    """
    # Ensure that the image is a Pillow Image object
    if not isinstance(image, pillow.Image.Image):
        raise ValueError("The input must be a Pillow Image object")
    
    # Check if the image has an alpha channel (RGBA mode)
    if image.mode == 'RGBA':
        # Create a white background in 'RGBA' mode
        background = pillow.Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Convert the original image to 'RGBA' mode if needed
        image = image.convert('RGBA')
        # Alpha composite to convert transparency to white
        image = pillow.Image.alpha_composite(background, image)
    else:
        # Convert other modes (like grayscale) to RGB
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Plot the image on the provided axes
    axs_img.imshow(image_array)
    
    # Get the original position of the axis
    pos = axs_img.get_position()
    new_pos = [pos.x0 + xoffset, pos.y0 + yoffset, pos.width * pltscale, pos.height * pltscale]
    
    # Adjust the position and scale
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
            pat_fr = bpf.create_grid_image(0,1.2)
            axs_pat.imshow(pat_fr)
        elif pr_no==1:
            axs_pat = axs_pat2  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(4,1.2)
            axs_pat.imshow(pat_fr)
        elif pr_no ==2:
            axs_pat = axs_pat3  #plt.subplot2grid((3,4),(0,p_no))
            pat_fr = bpf.create_grid_image(17,1.2)
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
        "X", "IX", "V", "IV",
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


def plot_raw_trace_time_points(single_cell_df,
                               deselect_list,fig,gs,
                               xoffset=0,yoffset=0):
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
                move_axis([axs_trace],xoffset=xoffset,yoffset=yoffset,pltscale=1)
                if pat_num==1:
                    axs_trace.set_ylabel("membrane potential(mV)")
                else:
                    axs_trace.set_ylabel(None)
                if pat_num ==0:
                    axs_trace.set_title("pre")
                    #axs_trace.text(-2,1.4,'B',transform=axs_trace.transAxes,    
                    #            fontsize=16, fontweight='bold', ha='center', va='center')            
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
                              alpha=0.6,label="pre training\nEPSP trace")
                axs_trace.plot(time,trace,
                               color=bpf.post_late,
                               label="post 30 mins of\ntraining EPSP trace")
                               #color=bpf.colorFader(bpf.post_color,
                               #                     bpf.post_late,
                               #                     (idx/len(pps_grp))))
                move_axis([axs_trace],xoffset=xoffset,yoffset=yoffset,pltscale=1)
                axs_trace.set_ylabel(None)
                axs_trace.set_yticklabels([])
                if pat_num==0:
                    if idx==1:
                        axs_trace.set_title(f"time points\n{time_points[idx+1]}")
                    else:
                        axs_trace.set_title(time_points[idx+1])
                else:
                    axs_trace.set_title(None)
            
            if (pat_num == 2) and (idx == 1):
                axs_trace.set_xlabel("time (ms)")
                
                # Collect handles and labels for the legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
            
                # Create custom legend handles with thicker lines
                custom_handles = [
                    Line2D([0], [0], color=handle.get_color(), linewidth=3, label=label) 
                    for label, handle in by_label.items()
                ]
            
                axs_trace.legend(custom_handles, by_label.keys(), 
                                 bbox_to_anchor=(0.1, -1.2),
                                 ncol=6,
                                 loc='center',
                                 frameon=False)




            #if (pat_num==2)and(idx==1):
            #    axs_trace.set_xlabel("time (ms)")
            #    handles, labels = plt.gca().get_legend_handles_labels()
            #    by_label = dict(zip(labels, handles))
            #    axs_trace.legend(by_label.values(), by_label.keys(), 
            #               bbox_to_anchor =(0.1, -0.95),
            #               ncol = 6,#title="cell response",
            #               loc='center',frameon=False)
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
                 

def plot_cell_type_features(cell_list, pattern_number, fig, axs_slp, val_to_plot, plt_color):
    if pattern_number == "pattern_0":
        pat_type = "trained"
    elif pattern_number == "pattern_1":
        pat_type = "overlapping"
    else:
        pat_type = "untrained"

    y_lim = (-50,700)
    time_points_= ["pre","0", "10", "20","30" ]
    pat_num = int(pattern_number.split("_")[-1])
    num_cells = len(cell_list["cell_ID"].unique())
    pfd = cell_list.groupby(by="frame_id")

    # Set a fixed y-axis limit for consistency across all plots
    axs_slp.set_ylim(y_lim)

    # Set a common base_y and step_y for all annotations
    base_y = 270  # Absolute y-axis position for the first annotation
    step_y = 40   # Spacing between each annotation

    # Define time points for x-tick labels
    time_point_list = ["pre", "post_0", "post_1", "post_2", "post_3"]

    for c, pat in pfd:
        if c != pattern_number:
            continue
        else:
            order = np.array(time_point_list, dtype=object)
            
            # Create the stripplot for individual cells (with legend label)
            stripplot = sns.stripplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}",
                order=order, ax=axs_slp, color=bpf.CB_color_cycle[2],
                alpha=0.6, size=5, label='single cell'
            )
            
            # Create the pointplot for the average of all cells (no automatic legend)
            sns.pointplot(
                data=pat, x="pre_post_status", y=f"{val_to_plot}",
                errorbar="se", order=order, capsize=0.1, ax=axs_slp,
                color=plt_color, linestyles='dotted', scale=0.8
            )
            
            # Create a custom marker for the point plot in the legend (instead of a line)
            pointplot_handle = Line2D([], [], marker='o', color=plt_color, linestyle='', label='average of\nall cells')

            # Set title, limits, and aesthetics
            stripplot.set_title(None)
            axs_slp.axhline(100, ls=':', color="k", alpha=0.4)
            stripplot.set(ylim=y_lim)

            # Set the custom x-ticks and labels based on time_points
            axs_slp.set_xticks(range(len(time_point_list)))  # Ensure correct positions for the labels
            axs_slp.set_xticklabels(time_points, rotation=30)  # Map time_points as x-tick labels

            # Collect p-values for annotations (if needed)
            pvalList = []
            anotp_list = []
            for i in order[1:]:
                posti = spst.wilcoxon(
                    pat[pat["pre_post_status"] == 'pre'][f"{val_to_plot}"],
                    pat[pat["pre_post_status"] == i][f"{val_to_plot}"],
                    zero_method="wilcox", correction=True
                )
                pvalList.append(posti.pvalue)
                anotp_list.append(("pre", i))

            # Manually add annotations using matplotlib
            for idx, (pval, pair) in enumerate(zip(pvalList, anotp_list)):
                x1, x2 = pair
                x1_pos = order.tolist().index(x1)
                x2_pos = order.tolist().index(x2)

                # Draw the annotation line
                axs_slp.plot([x1_pos, x2_pos], [base_y + idx * step_y] * 2, color='black', linewidth=1)

                # Add the p-value text above the line
                annotation_text = bpf.convert_pvalue_to_asterisks(pval)
                axs_slp.text(
                    (x1_pos + x2_pos) / 2, base_y + idx * step_y + 2, 
                    annotation_text, ha='center', va='bottom', fontsize=8
                )

            # Adjust axis labels and despine
            if pat_num == 0:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_ylabel("% change in\nEPSP amplitude")
                axs_slp.set_xlabel(None)
                # Remove legend for panel Di
                if axs_slp.get_legend() is not None:
                    axs_slp.get_legend().remove()
            elif pat_num == 1:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_ylabel(None)
                axs_slp.set_yticklabels([])
                axs_slp.set_xlabel("time points (mins)")
                # Remove legend for panel Dii
                if axs_slp.get_legend() is not None:
                    axs_slp.get_legend().remove()
            elif pat_num == 2:
                sns.despine(ax=axs_slp, top=True, right=True)
                axs_slp.set_xlabel(None)
                axs_slp.set_ylabel(None)

                # Get the handles and labels for the stripplot
                handles1, labels1 = stripplot.get_legend_handles_labels()

                # Remove duplicate "single cell" entries
                handles1_labels_dict = dict(zip(labels1, handles1))  # Ensure unique entries
                handles = list(handles1_labels_dict.values()) + [pointplot_handle]
                labels = list(handles1_labels_dict.keys()) + ['average of\nall cells']

                # Combine legends and add them to the plot
                axs_slp.legend(
                    handles, labels,
                    bbox_to_anchor=(0.8, 0.9),  # Adjust the legend position
                    ncol=1, title="cell response",
                    loc='upper center',
                    handletextpad=0.2,  # Reduce space between marker and text
                    labelspacing=0.2,  # Reduce vertical space between entries
                    fancybox=True,  # Enable fancy box with rounded corners
                    framealpha=0.7,  # Set the transparency of the legend background
                    facecolor='white'  # Set the background color to white
                )

    return axs_slp


#def plot_frequency_distribution(cell_list, val_to_plot, fig, ax,
#                                time_point='post_3', colors=None):
#    bins = 10 #np.linspace(-10, 10, num=10)
#
#    """
#    Plots a frequency distribution (histogram) for the specified time point,
#    comparing 'pattern_0', 'pattern_1', and 'pattern_2', and overlays a KDE.
#
#    Parameters:
#    - cell_list: pandas DataFrame containing the data
#    - val_to_plot: string, name of the column to plot
#    - fig: matplotlib Figure object
#    - ax: matplotlib Axes object
#    - time_point: string, the time point to filter on (default is 'post_3')
#    - bins: int or sequence, number of bins or bin edges for the histogram
#    - colors: list of colors for each pattern (default is None)
#
#    Returns:
#    - ax: matplotlib Axes object with the plot
#    """
#
#    # Define the patterns within the function
#    patterns = ['pattern_0', 'pattern_1', 'pattern_2']
#
#    # Run cell_list through norm_values before filtering the data
#    cell_list = norm_values(cell_list, val_to_plot)
#    norm_val_to_plot = f'{val_to_plot}'
#
#    # Check if colors are provided; if not, use default seaborn palette
#    if colors is None:
#        colors = sns.color_palette('Set1', n_colors=len(patterns))
#
#    # Loop over each pattern and plot the histogram with KDE
#    for pattern, color in zip(patterns, colors):
#        # Filter the data for the current pattern and time point
#        data_filtered = cell_list[
#            (cell_list['frame_id'] == pattern) &
#            (cell_list['pre_post_status'] == time_point)
#        ]
#
#        # Extract the normalized values to plot
#        values = data_filtered[norm_val_to_plot]
#
#        # Plot the histogram with KDE
#        sns.histplot(
#            values,
#            bins=bins,
#            kde=True,  # Set to True to overlay the KDE
#            label=pattern.replace('_', ' ').capitalize(),
#            color=color,
#            ax=ax,
#            alpha=0.6,
#            edgecolor='black'
#        )
#
#    # Customize the plot
#    ax.set_title(f'Frequency Distribution of {val_to_plot} at {time_point}')
#    ax.set_xlabel(f'Normalized {val_to_plot} (% of Pre)')
#    ax.set_ylabel('Frequency')
#    ax.legend(title='Patterns')
#    ax.set_xlim(-50,700)
#    sns.despine(ax=ax, trim=True)
#    #ax.set_xlim(bins[0], bins[-1])
#
#    return ax


def plot_frequency_distribution(cell_list, val_to_plot, fig, ax,
                                time_point='post_3', colors=None):
    """
    Plots a frequency distribution (histogram) for the specified time point,
    comparing 'trained', 'overlapping', and 'non-overlapping', and overlays a scaled KDE.
    Also performs Levene's test to compare variances between the groups.

    Parameters:
    - cell_list: pandas DataFrame containing the data
    - val_to_plot: string, name of the column to plot
    - fig: matplotlib Figure object
    - ax: matplotlib Axes object
    - time_point: string, the time point to filter on (default is 'post_3')
    - colors: list of colors for each pattern (default is None)

    Returns:
    - ax: matplotlib Axes object with the plot
    """

    # Define the patterns within the function
    patterns = ['pattern_0', 'pattern_1', 'pattern_2']

    # Mapping from pattern names to desired labels
    pattern_labels = {
        'pattern_0': 'trained',
        'pattern_1': 'overlapping',
        'pattern_2': 'non-overlapping'
    }

    # Run cell_list through norm_values before filtering the data
    cell_list = norm_values(cell_list, val_to_plot)
    norm_val_to_plot = f'{val_to_plot}'

    # Check if colors are provided; if not, use custom colors
    if colors is None:
        # Define custom colors that are not in the default plots
        colors = ['#00796B', '#D81B60', '#FFC107']
        #colors = ['#228B22', '#FF6F61', '#6A5ACD'] 
        #colors = ['#00796B','#FF6F61','#FFC107']
        #colors = ['#191970', '#FFD700', '#2E8B57']
        #colors = ['teal', 'orange', 'purple']
        #colors = ['#00BFFF', '#FA8072', '#808000']
        #colors = ['#4682B4', '#FFDAB9', '#9ACD32']
        #colors = ['#800000', '#FFBF00', '#228B22']
        #colors = ['#40E0D0', '#E6E6FA', '#CC5500']

    # Determine the x-axis limits as desired
    x_min, x_max = -50, 600  # Adjust these values as needed
    ax.set_xlim(x_min, x_max)
    data_list = cell_list[
        cell_list['frame_id'].str.contains("pattern") &
        (cell_list['pre_post_status'] == time_point)
    ][norm_val_to_plot].to_numpy()
    print(f"data_list:{data_list}")

    # Define the bins
    bins = int(np.sqrt(len(data_list)))

    # Initialize a dictionary to store values for Levene's test
    pattern_values = {}

    # Loop over each pattern and plot the histogram with KDE
    for pattern, color in zip(patterns, colors):
        # Filter the data for the current pattern and time point
        data_filtered = cell_list[
            (cell_list['frame_id'] == pattern) &
            (cell_list['pre_post_status'] == time_point)
        ]

        # Extract the normalized values to plot
        values = data_filtered[norm_val_to_plot]

        # Remove NaN and infinite values
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        # Store values for Levene's test
        pattern_values[pattern_labels[pattern]] = values

        # Plot the histogram with counts
        counts, bin_edges, patches = ax.hist(
            values,
            bins=bins,
            label=pattern_labels[pattern],  # Use the mapped label
            color=color,
            alpha=0.6,
            edgecolor=color,
            linewidth=0.5
        )

        # Compute the KDE
        if len(values) > 1:  # Need at least two data points for KDE
            kde = gaussian_kde(values)
            x_range = np.linspace(x_min, x_max, 1000)
            kde_values = kde.evaluate(x_range)

            # Scale the KDE to match counts
            bin_width = len(data_list)  # Using the same KDE estimation as in your original script
            scaled_kde = kde_values * len(values) * bin_width

            # Plot the scaled KDE
            ax.plot(x_range, scaled_kde, color=color)
        else:
            print(f"Not enough data points for KDE for {pattern}.")

    # Perform Levene's test between the three groups
    if all(len(v) > 1 for v in pattern_values.values()):
        levene_stat, levene_p = levene(*pattern_values.values())
        print(f"Levene's test statistic: {levene_stat}, p-value: {levene_p}")

        # Display the Levene's test p-value on the plot
        ax.text(0.16, 0.95, f"{bpf.convert_pvalue_to_asterisks(levene_p)}",
                transform=ax.transAxes,
                fontsize=10, va='top', ha='right')



        #ax.text(0.4, 1.1, f"Levene's p = {levene_p:.3f}",
        #        transform=ax.transAxes,
        #        fontsize=10, va='top', ha='right',
        #        bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    else:
        print("Not enough data points to perform Levene's test.")

    # Customize the plot
    ax.set_xlabel(f'% change in EPSP amplitude\npost 30 mins training')
    ax.set_ylabel('Number of cells')  # Since we're plotting counts
    ax.legend(title='Patterns')
    ax.text(0.3, 0.9, f"bin size: {bins}", transform=ax.transAxes,
            fontsize=10, va='top', ha='left')

    # Ensure x-axis limits are exactly as defined
    ax.set_xlim(x_min, x_max)

    # Remove top and right spines (similar to sns.despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax



#def plot_frequency_distribution(cell_list, val_to_plot, fig, ax,
#                                time_point='post_3', colors=None):
#    """
#    Plots a frequency distribution (histogram) for the specified time point,
#    comparing 'trained', 'overlapping', and 'non-overlapping', and overlays a scaled KDE.
#
#    Parameters:
#    - cell_list: pandas DataFrame containing the data
#    - val_to_plot: string, name of the column to plot
#    - fig: matplotlib Figure object
#    - ax: matplotlib Axes object
#    - time_point: string, the time point to filter on (default is 'post_3')
#    - colors: list of colors for each pattern (default is None)
#
#    Returns:
#    - ax: matplotlib Axes object with the plot
#    """
#
#    # Define the patterns within the function
#    patterns = ['pattern_0', 'pattern_1', 'pattern_2']
#
#    # Mapping from pattern names to desired labels
#    pattern_labels = {
#        'pattern_0': 'trained',
#        'pattern_1': 'overlapping',
#        'pattern_2': 'non-overlapping'
#    }
#
#    # Run cell_list through norm_values before filtering the data
#    cell_list = norm_values(cell_list, val_to_plot)
#    norm_val_to_plot = f'{val_to_plot}'
#
#    # Check if colors are provided; if not, use custom colors
#    if colors is None:
#        # Define custom colors that are not in the default plots
#        colors = ['teal', 'orange', 'purple']
#
#    # Determine the x-axis limits as desired
#    x_min, x_max = -50, 600  # Adjust these values as needed
#    ax.set_xlim(x_min, x_max)
#    data_list = cell_list[
#        cell_list['frame_id'].str.contains("pattern")&
#        (cell_list['pre_post_status'] == time_point)
#    ][norm_val_to_plot].to_numpy()
#    print(f"data_list:{data_list}")
#
#
#    # Define the bins
#    #bins = np.linspace(x_min, x_max, num=11)  # Adjust 'num' for number of bins
#    bins = int(np.sqrt(len(data_list)))
#    # Loop over each pattern and plot the histogram with KDE
#    for pattern, color in zip(patterns, colors):
#        # Filter the data for the current pattern and time point
#        data_filtered = cell_list[
#            (cell_list['frame_id'] == pattern) &
#            (cell_list['pre_post_status'] == time_point)
#        ]
#
#        # Extract the normalized values to plot
#        values = data_filtered[norm_val_to_plot]
#
#        # Remove NaN and infinite values
#        values = values.replace([np.inf, -np.inf], np.nan).dropna()
#
#        # Plot the histogram with counts
#        counts, bin_edges, patches = ax.hist(
#            values,
#            bins=bins,
#            label=pattern_labels[pattern],  # Use the mapped label
#            color=color,
#            alpha=0.6,
#            edgecolor='black',
#            linewidth=0.5
#        )
#
#        # Compute the KDE
#        if len(values) > 1:  # Need at least two data points for KDE
#            kde = gaussian_kde(values)
#            x_range = np.linspace(x_min, x_max, 1000)
#            kde_values = kde.evaluate(x_range)
#
#            # Scale the KDE to match counts
#            bin_width = len(data_list)
#            scaled_kde = kde_values * len(values) * bin_width
#
#            # Plot the scaled KDE
#            ax.plot(x_range, scaled_kde, color=color)
#        else:
#            print(f"Not enough data points for KDE for {pattern}.")
#
#    # Customize the plot
#    #ax.set_title(f'Frequency Distribution of {val_to_plot} at {time_point}')
#    ax.set_xlabel(f'% change in EPSP amplitude\npost 30 mins training')
#    ax.set_ylabel('Number of cells')  # Since we're plotting counts
#    ax.legend(title='Patterns')
#    ax.text(0.3,0.9,f"bin size: {bins}",transform=ax.transAxes,
#            fontsize=10, va='top', ha='left')
#
#    # Ensure x-axis limits are exactly as defined
#    ax.set_xlim(x_min, x_max)
#
#    # Remove top and right spines (similar to sns.despine)
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#
#    # Ensure x and y ticks are shown by commenting out this line
#    # ax.tick_params(axis='both', which='both', length=0)
#
#
#    return ax



#def plot_cell_type_features(cell_list, pattern_number, fig, axs_slp, val_to_plot, plt_color):
#    if pattern_number == "pattern_0":
#        pat_type = "trained"
#    elif pattern_number == "pattern_1":
#        pat_type = "overlapping"
#    else:
#        pat_type = "untrained"
#
#    y_lim = (-50, 700)
#    pat_num = int(pattern_number.split("_")[-1])
#    num_cells = len(cell_list["cell_ID"].unique())
#    pfd = cell_list.groupby(by="frame_id")
#    
#    for c, pat in pfd:
#        if c != pattern_number:
#            continue
#        else:
#            order = np.array(('pre', 'post_0', 'post_1', 'post_2', 'post_3'), dtype=object)
#            
#            # Create the stripplot for individual cells (with legend label)
#            sns.stripplot(
#                data=pat, x="pre_post_status", y=f"{val_to_plot}",
#                order=order, ax=axs_slp, color=bpf.CB_color_cycle[2],
#                alpha=0.6, size=5, label='single cell'
#            )
#            
#            # Create the pointplot for the average of all cells (no automatic legend)
#            sns.pointplot(
#                data=pat, x="pre_post_status", y=f"{val_to_plot}",
#                errorbar="se", order=order, capsize=0.1, ax=axs_slp,
#                color=plt_color, linestyles='dotted', scale=0.8
#            )
#            
#            # Create a custom marker for the point plot in the legend (instead of a line)
#            pointplot_handle = Line2D([], [], marker='o', color=plt_color, linestyle='', label='average of\nall cells')
#
#            # Set title and aesthetics
#            axs_slp.axhline(100, ls=':', color="k", alpha=0.4)
#            axs_slp.set_xticklabels(time_points, rotation=30)
#            
#            # Collect p-values for annotations (if needed)
#            pvalList = []
#            anotp_list = []
#            for i in order[1:]:
#                posti = spst.wilcoxon(
#                    pat[pat["pre_post_status"] == 'pre'][f"{val_to_plot}"],
#                    pat[pat["pre_post_status"] == i][f"{val_to_plot}"],
#                    zero_method="wilcox", correction=True
#                )
#                pvalList.append(posti.pvalue)
#                anotp_list.append(("pre", i))
#
#            # Annotate with p-values
#            annotator = Annotator(axs_slp, anotp_list, data=pat,
#                                  x="pre_post_status", y=f"{val_to_plot}",
#                                  order=order, fontsize=8)
#            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
#            annotator.annotate()
#
#            # Adjust axis labels and despine
#            if pat_num == 0:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_ylabel("% change in\nEPSP amplitude")
#                axs_slp.set_xlabel(None)
#            elif pat_num == 1:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_ylabel(None)
#                axs_slp.set_xlabel("time points (mins)")
#            elif pat_num == 2:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_xlabel(None)
#                axs_slp.set_ylabel(None)
#
#                # Get the handles and labels for the stripplot
#                handles1, labels1 = axs_slp.get_legend_handles_labels()
#
#                # Remove duplicate "single cell" entries
#                handles1_labels_dict = dict(zip(labels1, handles1))  # Ensure unique entries
#                handles = list(handles1_labels_dict.values()) + [pointplot_handle]
#                labels = list(handles1_labels_dict.keys()) + ['average of\nall cells']
#
#                # Combine legends and add them to the plot
#                axs_slp.legend(
#                    handles, labels,
#                    bbox_to_anchor=(0.8, 0.9),  # Adjust the legend position
#                    ncol=1, title="cell response",
#                    loc='upper center',
#                    handletextpad=0.2,  # Reduce space between marker and text
#                    labelspacing=0.2,  # Reduce vertical space between entries
#                    fancybox=True,  # Enable fancy box with rounded corners
#                    framealpha=0.7,  # Set the transparency of the legend background
#                    facecolor='white'  # Set the background color to white
#                )
#
#    # Set consistent y-axis limits after all plotting is done
#    axs_slp.set_ylim(y_lim)

#def plot_cell_type_features(cell_list, pattern_number, fig, axs_slp, val_to_plot, plt_color):
#    if pattern_number == "pattern_0":
#        pat_type = "trained"
#    elif pattern_number == "pattern_1":
#        pat_type = "overlapping"
#    else:
#        pat_type = "untrained"
#
#    y_lim = (-50, 500)
#    pat_num = int(pattern_number.split("_")[-1])
#    num_cells = len(cell_list["cell_ID"].unique())
#    pfd = cell_list.groupby(by="frame_id")
#    
#    for c, pat in pfd:
#        if c != pattern_number:
#            continue
#        else:
#            order = np.array(('pre', 'post_0', 'post_1', 'post_2', 'post_3'), dtype=object)
#            
#            # Create the stripplot for individual cells (with legend label)
#            stripplot = sns.stripplot(
#                data=pat, x="pre_post_status", y=f"{val_to_plot}",
#                order=order, ax=axs_slp, color=bpf.CB_color_cycle[2],
#                alpha=0.6, size=5, label='single cell'
#            )
#            
#            # Create the pointplot for the average of all cells (no automatic legend)
#            sns.pointplot(
#                data=pat, x="pre_post_status", y=f"{val_to_plot}",
#                errorbar="se", order=order, capsize=0.1, ax=axs_slp,
#                color=plt_color, linestyles='dotted', scale=0.8
#            )
#            
#            # Create a custom marker for the point plot in the legend (instead of a line)
#            pointplot_handle = Line2D([], [], marker='o', color=plt_color, linestyle='', label='average of\nall cells')
#
#            # Set title, limits, and aesthetics
#            stripplot.set_title(None)
#            axs_slp.axhline(100, ls=':', color="k", alpha=0.4)
#            stripplot.set(ylim=y_lim)
#            stripplot.set_xticklabels(time_points, rotation=30)
#            
#            # Collect p-values for annotations (if needed)
#            pvalList = []
#            anotp_list = []
#            for i in order[1:]:
#                posti = spst.wilcoxon(
#                    pat[pat["pre_post_status"] == 'pre'][f"{val_to_plot}"],
#                    pat[pat["pre_post_status"] == i][f"{val_to_plot}"],
#                    zero_method="wilcox", correction=True
#                )
#                pvalList.append(posti.pvalue)
#                anotp_list.append(("pre", i))
#
#            # Annotate with p-values
#            annotator = Annotator(axs_slp, anotp_list, data=pat,
#                                  x="pre_post_status", y=f"{val_to_plot}",
#                                  order=order, fontsize=8)
#            annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(a) for a in pvalList])
#            annotator.annotate()
#            # Adjust axis labels and despine
#            if pat_num == 0:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_ylabel("% change in\nEPSP amplitude")
#                axs_slp.set_xlabel(None)
#            elif pat_num == 1:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_ylabel(None)
#                axs_slp.set_xlabel("time points (mins)")
#            elif pat_num == 2:
#                sns.despine(fig=None, ax=axs_slp, top=True, right=True)
#                axs_slp.set_xlabel(None)
#                axs_slp.set_ylabel(None)
#
#                # Get the handles and labels for the stripplot
#                handles1, labels1 = stripplot.get_legend_handles_labels()
#
#                # Remove duplicate "single cell" entries
#                handles1_labels_dict = dict(zip(labels1, handles1))  # Ensure unique entries
#                handles = list(handles1_labels_dict.values()) + [pointplot_handle]
#                labels = list(handles1_labels_dict.keys()) + ['average of\nall cells']
#
#                # Combine legends and add them to the plot
#                axs_slp.legend(
#                    handles, labels,
#                    bbox_to_anchor=(0.8, 0.9),  # Adjust the legend position
#                    ncol=1, title="cell response",
#                    loc='upper center',
#                    handletextpad=0.2,  # Reduce space between marker and text
#                    labelspacing=0.2,  # Reduce vertical space between entries
#                    fancybox=True,  # Enable fancy box with rounded corners
#                    framealpha=0.7,  # Set the transparency of the legend background
#                    facecolor='white'  # Set the background color to white
#                )
#
#            else:
#                pass


            
        
            
def plot_field_normalised_feature_multi_patterns(cell_list,val_to_plot,
                                                fig,axs1,axs2,axs3):
    cell_list= norm_values(cell_list,val_to_plot)
    plot_cell_type_features(cell_list,"pattern_0",fig, axs1,val_to_plot,
                            bpf.CB_color_cycle[5])
    plot_cell_type_features(cell_list,"pattern_1",fig, axs2,val_to_plot,
                            bpf.CB_color_cycle[5])
    plot_cell_type_features(cell_list,"pattern_2",fig, axs3,val_to_plot,
                            bpf.CB_color_cycle[5])


def plot_field_normalised_feature_multi_patterns_with_stats(cell_list, val_to_plot, fig, axs1, axs2, axs3, stat_method="wilcoxon"):
    """
    Modified version that uses different statistical tests
    """
    cell_list = norm_values(cell_list, val_to_plot)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_0", fig, axs1, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_1", fig, axs2, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)
    plot_cell_type_features_with_custom_stats(cell_list, "pattern_2", fig, axs3, val_to_plot,
                                            bpf.CB_color_cycle[5], stat_method)
    

#["cell_ID","frame_status","pre_post_status","frame_id","min_trace","max_trace","abs_area","pos_area",
#"neg_area","onset_time","max_field","min_field","slope","intercept","min_trace_t","max_trace_t","max_field_t","min_field_t","mean_trace","mean_field","mean_ttl","mean_rmp"]


def inR_sag_plot(inR_all_Cells_df, fig, axs):
    deselect_list = ['post_4', 'post_5']
    inR_all_Cells_df = inR_all_Cells_df[~inR_all_Cells_df["pre_post_status"].isin(deselect_list)]
    order = np.array(('pre', 'post_0', 'post_1', 'post_2', 'post_3'), dtype=object)

    # Plot input resistance and sag values using pointplot
    g1 = sns.pointplot(data=inR_all_Cells_df, x="pre_post_status", y="inR",
                       capsize=0.2, ci='sd', order=order, color="k")
    g2 = sns.pointplot(data=inR_all_Cells_df, x="pre_post_status", y="sag",
                       capsize=0.2, ci='sd', order=order, color=bpf.CB_color_cycle[4])
    
    # Plot individual points using stripplot
    sns.stripplot(data=inR_all_Cells_df, color=bpf.CB_color_cycle[4],
                  x="pre_post_status", y="sag",
                  order=order, alpha=0.2)

    # Perform Wilcoxon test between "pre" and "post_3"
    pre_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"] == "pre"]["sag"]
    post_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"] == "post_3"]["sag"]
    pre = spst.wilcoxon(pre_trace, post_trace, zero_method="wilcox", correction=True)
    pvalList = pre.pvalue
    print(f"p-value: {pvalList}")
    anotp_list = ("pre", "post_3")
    annotator = Annotator(axs, [anotp_list], data=inR_all_Cells_df, x="pre_post_status", y="sag", order=order)
    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
    annotator.annotate()

    # Manually create legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='k', label='input resistance', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color=bpf.CB_color_cycle[4], label='sag value', markersize=8, linestyle='None')
    ]

    #legend_elements = [
    #    Line2D([0], [0], color='k', label='input resistance', linewidth=2),
    #    Line2D([0], [0], color=bpf.CB_color_cycle[4], label='sag value', linewidth=2)
    #]
    axs.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.05), loc='center', frameon=False)

    # Set axis labels, ticks, and limits
    axs.set_ylabel("MOhms")
    axs.set_xlabel("time points (mins)")

    # Set x-tick labels using the original time_points variable
    time_points = ['pre', '0', '10', '20', '30']
    axs.set_xticklabels(time_points)

    axs.set_ylim(-10, 250)
    sns.despine(fig=None, ax=axs, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    # Adjust the position of the axis
    inr_pos = axs.get_position()
    new_inr_pos = [inr_pos.x0, inr_pos.y0 - 0.04, inr_pos.width, inr_pos.height]
    axs.set_position(new_inr_pos)


#def inR_sag_plot(inR_all_Cells_df,fig,axs):
#    deselect_list = ['post_4','post_5']
#    inR_all_Cells_df =inR_all_Cells_df[~inR_all_Cells_df["pre_post_status"].isin(deselect_list)] 
#    order = np.array(('pre','post_0', 'post_1', 'post_2', 'post_3'),dtype=object)
#
#    g=sns.pointplot(data=inR_all_Cells_df,x="pre_post_status",y="inR",
#                    capsize=0.2,ci=('sd'),order=order,color="k",
#                    label="input\nresistance")
#    sns.pointplot(data=inR_all_Cells_df,x="pre_post_status",y="sag",
#                  capsize=0.2,ci=('sd'),order=order,
#                  color=bpf.CB_color_cycle[4], label="sag value")
#    sns.stripplot(data=inR_all_Cells_df,color=bpf.CB_color_cycle[4],
#                  x="pre_post_status",y="sag",
#                  order=order,alpha=0.2)
#
#    pre_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"]=="pre"]["sag"]
#    post_trace = inR_all_Cells_df[inR_all_Cells_df["pre_post_status"]=="post_3"]["sag"]
#    pre= spst.wilcoxon(pre_trace,post_trace,zero_method="wilcox", correction=True)
#    pvalList=pre.pvalue
#    print(pvalList)
#    anotp_list=("pre","post_3")
#    annotator = Annotator(axs,[anotp_list],data=inR_all_Cells_df, x="pre_post_status",y="sag",order=order)
#    #annotator = Annotator(axs[pat_num],[("pre","post_0"),("pre","post_1"),("pre","post_2"),("pre","post_3")],data=cell, x="pre_post_status",y=f"{col_pl}")
#    annotator.set_custom_annotations([bpf.convert_pvalue_to_asterisks(pvalList)])
#    annotator.annotate()
#    if axs.legend_ is not None:
#        axs.legend_.remove()
#
#    #sns.move_legend(axs, "upper left", bbox_to_anchor=(1, 1))
#    #axs.set_ylim(-10,250)
#    axs.set_xticklabels(time_points)
#    sns.despine(fig=None, ax=axs, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
#    axs.set_ylabel("MOhms")
#    axs.set_xlabel("time points (mins)")
#    handles, labels = g.get_legend_handles_labels()
#    by_label = dict(zip(labels, handles))
#    axs.legend(by_label.values(), by_label.keys(), 
#                   bbox_to_anchor =(0.5, 1.275),
#                   ncol = 2,
#                   loc='upper center',frameon=False)
#
#
#
#
#    inr_pos = axs.get_position()
#    new_inr_pos = [inr_pos.x0, inr_pos.y0-0.04, inr_pos.width,
#                   inr_pos.height]
#    axs.set_position(new_inr_pos)
    
    

def plot_figure_2_with_stats(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  inR_all_Cells_df,
                  illustration_path,
                  inRillustration_path,
                  patillustration_path,
                  outdir,cell_to_plot=cell_to_plot,
                  stat_method="wilcoxon"):
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
    patillustration = pillow.Image.open(patillustration_path)
    # Define the width and height ratios
    width_ratios = [1, 1, 1, 1, 1, 1, 0.8]  # Adjust these values as needed
    height_ratios = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 
                     0.5, 0.5, 0.5, 0.5, 0.2]       # Adjust these values as needed

    fig = plt.figure(figsize=(9,16))
    gs = GridSpec(11, 7,width_ratios=width_ratios,
                  height_ratios=height_ratios,figure=fig)
    #gs.update(wspace=0.2, hspace=0.8)
    gs.update(wspace=0.2, hspace=0.2)
    #place illustration
    axs_img = fig.add_subplot(gs[:3, :6])
    plot_image(illustration,axs_img,-0.1,-0.01,1)
    
    

    bpf.add_subplot_label(axs_img, 'A', xpos=0, ypos=0.95, fontsize=16, fontweight='bold', ha='center', va='center')
    
    axs_pat_ill = fig.add_subplot(gs[:3, 6:])
    plot_image(patillustration,axs_pat_ill,-0.15,-0.02,2.2)    
    
    bpf.add_subplot_label(axs_pat_ill, 'B', xpos=0.01, ypos=0.9, fontsize=16, fontweight='bold', ha='center', va='center')

    axs_vpat1=fig.add_subplot(gs[3,0])
    axs_vpat2=fig.add_subplot(gs[4,0])
    axs_vpat3=fig.add_subplot(gs[5,0])
    plot_patterns(axs_vpat1,axs_vpat2,axs_vpat3,-0.075,-0.015,2)
    bpf.add_subplot_label(axs_vpat1, 'C', xpos=-0.07, ypos=1.35, fontsize=16, fontweight='bold', ha='center', va='center')

    plot_raw_trace_time_points(single_cell_df,deselect_list,fig,gs,
                              xoffset=0,yoffset=-0.01)
    
    
    
    
    
    
    
    

    #plot amplitudes over time
    feature_extracted_data =feature_extracted_data[~feature_extracted_data["frame_status"].isin(deselect_list)]
    cell_grp = feature_extracted_data.groupby(by="cell_ID")
    axs_slp1 = fig.add_subplot(gs[7:9,0:2])
    axs_slp1.set_ylabel("slope (mV/ms)")
    axs_slp2 = fig.add_subplot(gs[7:9,2:4])
    axs_slp2.set_yticklabels([])
    axs_slp3 = fig.add_subplot(gs[7:9,4:6])
    axs_slp3.set_yticklabels([])
    plot_field_normalised_feature_multi_patterns_with_stats(sc_data_df,"max_trace",
                                                 fig,axs_slp1,axs_slp2,
                                                 axs_slp3, stat_method)
    axs_slp_list = [axs_slp1,axs_slp2,axs_slp3]
    d_axes = axs_slp_list
    d_labels = bpf.generate_letter_roman_labels('D', len(d_axes))
    bpf.add_subplot_labels_from_list(d_axes, d_labels, 
                                base_params={'xpos': -0.1, 'ypos': 1.1, 'fontsize': 16, 'fontweight': 'bold'})

    #plot pattern projections 
    axs_pat1 = fig.add_subplot(gs[6,0])
    axs_pat2 = fig.add_subplot(gs[6,2])
    axs_pat3 = fig.add_subplot(gs[6,4])
    plot_patterns(axs_pat1,axs_pat2,axs_pat3,0.05,-0.055,1)





    axs_dist = fig.add_subplot(gs[9:10,0:4])
    plot_frequency_distribution(sc_data_df, "max_trace", fig, 
                                axs_dist, time_point='post_3', 
                                colors=None)
    bpf.add_subplot_label(axs_dist, 'E', xpos=-0.07, ypos=1, fontsize=16, fontweight='bold', ha='center', va='center')
    move_axis([axs_dist],xoffset=0,yoffset=-0.03,pltscale=1)
    
    #axs_inr = fig.add_subplot(gs[9:10,3:6])
    #inR_sag_plot(inR_all_Cells_df,fig,axs_inr)
    #axs_inr.text(-0.05,1,'F',transform=axs_inr.transAxes,    
    #         fontsize=16, fontweight='bold', ha='center', va='center')            


    #axs_inrill = fig.add_subplot(gs[9:10,0:3])
    #plot_image(inRillustration,axs_inrill,-0.05,-0.05,1)
    #axs_inrill.text(0.01,1.1,'E',transform=axs_inrill.transAxes,    
    #             fontsize=16, fontweight='bold', ha='center', va='center')            


    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    
    # Set filename based on statistical method
    if stat_method == "wilcoxon":
        filename = "figure_2_wilcox_sr_test.png"
    elif stat_method == "rm_anova":
        filename = "figure_2_rep_mesure_anova.png"
    elif stat_method == "mixed_effect":
        filename = "figure_2_mixd_effect_model.png"
    else:
        filename = "figure_2.png"
    
    outpath = f"{outdir}/{filename}"
    plt.savefig(outpath,bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def plot_figure_2(extracted_feature_pickle_file_path,
                  cell_categorised_pickle_file,
                  inR_all_Cells_df,
                  illustration_path,
                  inRillustration_path,
                  patillustration_path,
                  outdir,cell_to_plot=cell_to_plot):
    """
    Generate all three versions of Figure 2 with different statistical tests
    """
    print("Generating Figure 2 with Wilcoxon signed-rank test...")
    plot_figure_2_with_stats(extracted_feature_pickle_file_path,
                            cell_categorised_pickle_file,
                            inR_all_Cells_df,
                            illustration_path,
                            inRillustration_path,
                            patillustration_path,
                            outdir, cell_to_plot, "wilcoxon")
    
    print("Generating Figure 2 with repeated measures ANOVA...")
    plot_figure_2_with_stats(extracted_feature_pickle_file_path,
                            cell_categorised_pickle_file,
                            inR_all_Cells_df,
                            illustration_path,
                            inRillustration_path,
                            patillustration_path,
                            outdir, cell_to_plot, "rm_anova")
    
    print("Generating Figure 2 with mixed effect model...")
    plot_figure_2_with_stats(extracted_feature_pickle_file_path,
                            cell_categorised_pickle_file,
                            inR_all_Cells_df,
                            illustration_path,
                            inRillustration_path,
                            patillustration_path,
                            outdir, cell_to_plot, "mixed_effect")
    
    print("All three versions of Figure 2 have been generated!")


def main():
    # Argument parser.
    description = '''EPSP Amplitude Analysis - Generates three versions of Figure 2 with different statistical tests for D panels'''
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
    parser.add_argument('--patillustration-path', '-m'
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
    patillustration_path =Path(args.patillustration_path) 
    inRillustration_path = Path(args.inRillustration_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_2'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_2(pklpath,scpath,inR_path,illustration_path,
                  inRillustration_path,
                  patillustration_path,
                  globoutdir)
    print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
