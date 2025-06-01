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
from shared_utils import baisic_plot_fuctnions_and_features as bpf
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon, permutation_test, ks_2samp
import multiprocessing as mp

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

def plot_image(image,axs_img,xoffset,yoffset,pltscale):
    axs_img.imshow(image, cmap='gray')
    pos = axs_img.get_position()  # Get the original position
    new_pos = [pos.x0+xoffset, pos.y0+yoffset, pos.width*pltscale,
               pos.height*pltscale]
    # Shrink the plot
    axs_img.set_position(new_pos)
    axs_img.axis('off')

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

def gamma_fit(expt, gamma):
    """
    Gamma function fit model with fixed parameters:
    beta = 1, alpha = 0.
    """
    beta = 1  # Fixed value
    alpha = 0  # Fixed value
    return expt - (((beta * expt) / (gamma + expt)) * expt) - alpha

def fitGamma( data ):
    data = data.transpose()
    [ret], cov = scipy.optimize.curve_fit(gamma_fit, 
        data[0], data[1], p0 = [2.0], bounds=(0, 60))
    return ret



# Define the worker function at the top level
def bootstrap_worker(args):
    combined, len_A = args
    idx = np.arange(len(combined))
    
    # Resample with replacement
    temp = np.random.choice(idx, size=len_A, replace=True)
    A_boot = combined[temp]
    temp = np.random.choice(idx, size=len_A, replace=True)
    B_boot = combined[temp]
    
    # Compute test statistic for resampled data
    return fitGamma(A_boot) - fitGamma(B_boot)

# Main bootstrap function
def bootstrap_scram(A, B, nIter=10000):
    # Convert list to array
    A = np.array(A)
    B = np.array(B)

    # Combine datasets
    combined = np.concatenate([A, B])

    # Calculate observed test statistic (e.g., difference in means)
    observed_stat = fitGamma(A) - fitGamma(B)

    # Prepare arguments for the worker function
    args = (combined, len(A))

    # Create a pool of workers
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        # Use pool.imap to distribute the computation across processes
        bootstrap_stats = list(tqdm(pool.imap(bootstrap_worker, [args] * nIter), total=nIter, desc="Bootstrapping"))

    # Convert to numpy array
    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))

    return p_value



#def bootstrap_scram(A,B, nIter=10000):
#    #corrected chatGPT code from Upi v2
#    #convert list to array
#    A=np.array(A)
#    B=np.array(B)
#
#    # Combine datasets
#    combined = np.concatenate([A, B])
#
#    # Calculate observed test statistic (e.g., difference in means)
#    observed_stat = fitGamma(A) - fitGamma(B)
#
#    # Bootstrap resampling
#    bootstrap_stats = []
#    idx = np.arange(len(combined))
#    for _ in tqdm(range(nIter), desc="Bootstrapping"):
#    #for _ in range(nIter):
#        # Resample with replacement 
#        temp = np.random.choice(idx, size=len(A), replace=True)
#        A_boot = combined[temp]
#        temp = np.random.choice(idx, size=len(A), replace=True)
#        B_boot = combined[temp]
#        '''
#        A_boot = np.random.choice(combined, size=len(A), replace=True)
#        B_boot = np.random.choice(combined, size=len(B), replace=True)
#        '''
#
#        # Compute test statistic for resampled data
#        bootstrap_stats.append(fitGamma(A_boot) - fitGamma(B_boot))
#
#    # Convert to numpy array
#    #bootstrap_stats = np.array(bootstrap_stats)
#
#    # Calculate p-value (two-tailed)
#    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
#
#    return p_value











#def bootstrap_scram(A, B, nIter=10000):
#    #chatgpt bootsratp from upi
#    #chatBootstrap(A, B, nIter=1000):
#    # Combine datasets
#    A=np.array(A)
#    B=np.array(B)
#    combined = np.concatenate([A, B])
#
#    # Calculate observed test statistic (e.g., difference in means)
#    observed_stat = fitGamma(A) - fitGamma(B)
#
#    # Bootstrap resampling
#    bootstrap_stats = []
#    idx = np.arange(len(A))
#    for _ in range(nIter):
#        # Resample with replacement 
#        temp = np.random.choice(idx, size=len(A), replace=True)
#        A_boot = combined[temp]
#        temp = np.random.choice(idx, size=len(A), replace=True)
#        B_boot = combined[temp]
#        '''
#        A_boot = np.random.choice(combined, size=len(A), replace=True)
#        B_boot = np.random.choice(combined, size=len(B), replace=True)
#        '''
#
#        # Compute test statistic for resampled data
#        bootstrap_stats.append(fitGamma(A_boot) - fitGamma(B_boot))
#
#    # Convert to numpy array
#    bootstrap_stats = np.array(bootstrap_stats)
#
#    # Calculate p-value (two-tailed)
#    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
#
#    return p_value















#def bootstrap_scram(list_of_x_y_responses_pre,
#                    list_of_x_y_responses_post,pat_num,cell_type,
#                   color):
#    #print(f"list_of_x_y_responses_pre:{list_of_x_y_responses_pre}..................")
#    list_of_x_y_responses_pre= np.array(list_of_x_y_responses_pre)
#    list_of_x_y_responses_post = np.array(list_of_x_y_responses_post)
#    pre_gama_dist=[]
#    post_gama_dist = []
#    for i in range(1000):
#        rng_pre = np.random.default_rng()
#        rng_post = np.random.default_rng()
#        pre_x_y = rng_pre.choice(list_of_x_y_responses_pre,
#                                 size=int(0.8*len(list_of_x_y_responses_pre)),
#                                 replace=False)
#        post_x_y_responses = rng_post.choice(list_of_x_y_responses_post,
#                                             size=int(0.8*len(list_of_x_y_responses_post)),
#                                             replace=False)
#
#        # Split the arrays into x (input) and y (response) parts
#        pre_arr1, pre_arr2 = np.split(pre_x_y, 2, axis=1)
#        arr1, arr2 = np.split(post_x_y_responses, 2, axis=1)
#        
#        # Flatten the arrays
#        pre_arr1 = np.ravel(pre_arr1)
#        pre_arr2 = np.ravel(pre_arr2)
#        arr1 = np.ravel(arr1)
#        arr2 = np.ravel(arr2)
#        
#        # Perform curve fitting only for the 'gam' parameter
#        param_pre, _ = scipy.optimize.curve_fit(gama_fit, pre_arr1, pre_arr2, bounds=(0, 60))
#        param_post, _ = scipy.optimize.curve_fit(gama_fit, arr1, arr2, bounds=(0, 60)
#                                                )
#        pre_gama_dist.append(param_pre[0])
#        post_gama_dist.append(param_post[0])
#
#    _, pvalue = ttest_rel(pre_gama_dist,post_gama_dist)
#    #_, pvalue =mannwhitneyu(pre_gama_dist,post_gama_dist)
#    #pvalue = bpf.convert_pvalue_to_asterisks(pvalue)
#    fig1,axs = plt.subplots(1,1,figsize=(12,9))
#    axs.set_title(f"pattern_{pat_num}_{cell_type}")
#    axs.hist(pre_gama_dist,bins=int(np.sqrt(len(pre_gama_dist))),color='k')
#    axs.hist(post_gama_dist, bins=int(np.sqrt(len(pre_gama_dist))),color=color)
#    axs.set_xlabel("gama")
#    axs.set_ylabel("#")
#    fig1.savefig(f"/Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/plotting_scripts/Figure_7/plt_dist_{pat_num}_{cell_type}.png")
#    print(f"p val = {pvalue}****************************")
#    plt.close(fig1)
#    return pvalue







def gama_fit(expt, gam):
    """
    Gamma function fit model with fixed parameters:
    bet = 1, alp = 0.
    """
    bet = 1  # Fixed value
    alp = 0  # Fixed value
    return expt - (((bet * expt) / (gam + expt)) * expt) - alp

def perform_statistical_tests(pre_y, y):
    """
    Perform statistical tests on the fitted y-values from pre and post conditions.
    """
    # Paired T-test
    t_stat, p_value_ttest = ttest_rel(pre_y, y)
    
    # Mann-Whitney U test
    u_stat, p_value_mannwhitney = mannwhitneyu(pre_y, y, alternative='two-sided')
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_value_wilcoxon = wilcoxon(pre_y, y)
    except ValueError:
        p_value_wilcoxon = np.nan  # Handle the case where the sample size is too small

    # Permutation test
    def custom_statistic(x, y):
        return np.mean(x) - np.mean(y)
    p_value_permutation = permutation_test((pre_y, y), custom_statistic, n_resamples=10000, alternative='two-sided').pvalue

    # Kolmogorov-Smirnov test
    ks_stat, p_value_ks = ks_2samp(pre_y, y)

    return {
        "Paired T-test": p_value_ttest,
        "Mann-Whitney U test": p_value_mannwhitney,
        "Wilcoxon test": p_value_wilcoxon,
        "Permutation test": p_value_permutation,
        "KS test": p_value_ks
    }

def bootstrap_diff(data1, data2, num_iterations=1000):
    """
    Perform bootstrapping to estimate confidence intervals for the difference between two datasets.
    """
    differences = []
    for _ in range(num_iterations):
        resample1 = np.random.choice(data1, size=len(data1), replace=True)
        resample2 = np.random.choice(data2, size=len(data2), replace=True)
        differences.append(np.mean(resample1) - np.mean(resample2))
    return np.percentile(differences, [2.5, 97.5])

def eq_fit(list_of_x_y_responses_pre, list_of_x_y_responses, pat_num, cell_type, fig, axs):
    """
    Fits the gamma function model to pre-training and post-training data,
    performs statistical tests on fitted y-values, and plots the results.
    """
    # Choose color based on cell type using the bpf module
    color = bpf.CB_color_cycle[0] if cell_type == "learners" else bpf.CB_color_cycle[1]

    # Convert input lists to numpy arrays
    pre_x_y = np.array(list_of_x_y_responses_pre)
    x_y_responses = np.array(list_of_x_y_responses)
    
    # Split the arrays into x (input) and y (response) parts
    pre_arr1, pre_arr2 = np.split(pre_x_y, 2, axis=1)
    arr1, arr2 = np.split(x_y_responses, 2, axis=1)

    # Flatten the arrays
    pre_arr1 = np.ravel(pre_arr1)
    pre_arr2 = np.ravel(pre_arr2)
    arr1 = np.ravel(arr1)
    arr2 = np.ravel(arr2)

    # Perform curve fitting only for the 'gam' parameter
    param_pre, _ = scipy.optimize.curve_fit(gama_fit, pre_arr1, pre_arr2, bounds=(0, 60))
    param_post, _ = scipy.optimize.curve_fit(gama_fit, arr1, arr2, bounds=(0, 60))

    # Generate x-values for plotting
    pre_x = np.linspace(-0.5, 10, len(pre_arr2))
    x = np.linspace(-0.5, 10, len(arr2))

    test_results = bootstrap_scram(list_of_x_y_responses_pre,
                                   list_of_x_y_responses)
                                   #,pat_num, 
                                   #cell_type, color)
    test_results= bpf.convert_pvalue_to_asterisks(test_results)
    # Generate fitted y-values using the optimized 'gam' parameter
    pre_y = gama_fit(pre_x, param_pre[0])
    y = gama_fit(x, param_post[0])

    ## Perform statistical tests on the fitted y-values
    #test_results = perform_statistical_tests(pre_y, y)

    ## Perform bootstrapping to get confidence intervals for the difference
    #ci_bootstrap = bootstrap_diff(pre_y, y)
    
    
    

    # Plot the fitted curves
    axs.plot(pre_x, pre_y, color='k', linestyle='-', alpha=0.8, label="pre_training", linewidth=3)
    axs.plot(x, y, color=color, linestyle='-', alpha=0.8, label="post_training", linewidth=3)

    # Display the fitted gamma values and test results on the plot
    axs.set_aspect(0.6)
    axs.text(0.5, 0.9, f'γ_post = {np.around(param_post[0], 2)}', transform=axs.transAxes, fontsize=12, ha='center')
    axs.text(0.5, 0.7, f'γ_pre = {np.around(param_pre[0], 2)}', transform=axs.transAxes, fontsize=12, ha='center')
    
    ## Display statistical test results with p-values
    axs.text(0.12, 0.8, f'{test_results}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.7, f'T-test p = {test_results["Paired T-test"]:.4f}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.6, f'MWU p = {test_results["Mann-Whitney U test"]:.4f}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.5, f'Wilcoxon p = {test_results["Wilcoxon test"]:.4f}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.4, f'Perm Test p = {test_results["Permutation test"]:.4f}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.3, f'KS Test p = {test_results["KS test"]:.4f}', transform=axs.transAxes, fontsize=12, ha='center')
    #axs.text(0.5, 0.2, f'Bootstrap CI: [{ci_bootstrap[0]:.2f}, {ci_bootstrap[1]:.2f}]', transform=axs.transAxes, fontsize=12, ha='center')

    return param_pre[0], param_post[0], test_results




















##def gama_fit(expt,alp,bet,gam):
##    return expt-(((bet*expt)/(gam+expt))*expt)-alp
#
#def gama_fit(expt, gam):
#    """
#    Gamma function fit model with fixed parameters:
#    bet = 1, alp = 0.
#    """
#    bet = 1  # Fixed value
#    alp = 0  # Fixed value
#    return expt - (((bet * expt) / (gam + expt)) * expt) - alp
#
#def safe_t_test(gamma_pre, gamma_post):
#    """
#    Performs a paired t-test safely, handling cases where p-values might become NaN.
#    """
#    if len(gamma_pre) < 2 or len(gamma_post) < 2:
#        print("Insufficient data for statistical testing.")
#        return np.nan
#
#    if np.allclose(gamma_pre, gamma_post):
#        print("No variability between pre and post; returning NaN.")
#        return np.nan
#
#    if np.isnan(gamma_pre).any() or np.isnan(gamma_post).any() or \
#       np.isinf(gamma_pre).any() or np.isinf(gamma_post).any():
#        print("Invalid data (NaN or inf) detected.")
#        return np.nan
#
#    t_stat, p_value = ttest_rel(gamma_pre, gamma_post)
#    return p_value
#
#def extract_single_value(df, condition, column):
#    """
#    Extracts a single value from a pandas DataFrame based on a condition.
#    Handles cases where the result might be an array.
#    """
#    values = df[condition][column].values
#    if values.size > 0:
#        return float(values[0])
#    else:
#        print("Warning: No matching value found.")
#        return np.nan
#
#def eq_fit(list_of_x_y_responses_pre, list_of_x_y_responses, pat_num, cell_type, fig, axs):
#    """
#    Fits the gamma function model to pre-training and post-training data,
#    performs statistical tests, and plots the results on the provided axes.
#    """
#    # Convert input lists to numpy arrays
#    pre_x_y = np.array(list_of_x_y_responses_pre)
#    x_y_responses = np.array(list_of_x_y_responses)
#    
#    # Split the arrays into x (input) and y (response) parts
#    pre_arr1, pre_arr2 = np.split(pre_x_y, 2, axis=1)
#    arr1, arr2 = np.split(x_y_responses, 2, axis=1)
#
#    # Flatten the arrays
#    pre_arr1 = np.ravel(pre_arr1)
#    pre_arr2 = np.ravel(pre_arr2)
#    arr1 = np.ravel(arr1)
#    arr2 = np.ravel(arr2)
#
#    # Perform curve fitting for the 'gam' parameter
#    try:
#        param_pre, _ = scipy.optimize.curve_fit(gama_fit, pre_arr1, pre_arr2, bounds=(0, 60))
#        param_post, _ = scipy.optimize.curve_fit(gama_fit, arr1, arr2, bounds=(0, 60))
#    except RuntimeError as e:
#        print(f"Curve fitting failed: {e}")
#        return None
#
#    # Extract the fitted gamma values
#    gamma_pre = [param_pre[0]]
#    gamma_post = [param_post[0]]
#    print(f"gamma_pre:{gamma_pre},gamma_post:{gamma_post}")
#    # Perform statistical comparison using the safe t-test function
#    p_value = safe_t_test(param_pre, param_post)
#
#    # Generate x-values for plotting
#    pre_x = np.linspace(-0.5, 10, len(pre_arr2))
#    x = np.linspace(-0.5, 10, len(arr2))
#
#    # Generate fitted y-values using the optimized 'gam' parameter
#    pre_y = gama_fit(pre_x, gamma_pre[0])
#    y = gama_fit(x, gamma_post[0])
#
#    # Choose color based on cell type
#    color = 'blue' if cell_type == "learners" else 'orange'
#
#    # Plot the fitted curves
#    axs.plot(pre_x, pre_y, color='k', linestyle='-', alpha=0.8, label="pre_training", linewidth=3)
#    axs.plot(x, y, color=color, linestyle='-', alpha=0.8, label="post_training", linewidth=3)
#
#    # Set plot properties and display fitted gamma values
#    axs.set_aspect(0.6)
#    axs.text(0.5, 0.9, f'γ_post = {np.around(gamma_post[0], 1)}', transform=axs.transAxes, fontsize=12, ha='center')
#    axs.text(0.5, 0.7, f'γ_pre = {np.around(gamma_pre[0], 1)}', transform=axs.transAxes, fontsize=12, ha='center')
#
#    # Display the p-value on the plot
#    if p_value is not None and not np.isnan(p_value):
#        if p_value < 0.05:
#            significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else "*")
#            axs.text(0.5, 0.5, f'p = {p_value:.3f} {significance}', transform=axs.transAxes, fontsize=12, ha='center', color='red')
#        else:
#            axs.text(0.5, 0.5, f'p = {p_value:.3f} (n.s.)', transform=axs.transAxes, fontsize=12, ha='center')
#    else:
#        axs.text(0.5, 0.5, 'p = NaN (check data)', transform=axs.transAxes, fontsize=12, ha='center', color='red')
#
#    return gamma_pre[0], gamma_post[0], p_value


##plot only gama 
#def gama_fit(expt, gam):
#    """
#    Gamma function fit model with fixed parameters:
#    bet = 1, alp = 0.
#    """
#    bet = 1  # Fixed value
#    alp = 0  # Fixed value
#    return expt - (((bet * expt) / (gam + expt)) * expt) - alp
#
#def eq_fit(list_of_x_y_responses_pre, list_of_x_y_responses, pat_num, cell_type, fig, axs):
#    """
#    Fits the gamma function model to pre-training and post-training data,
#    and plots the results on the provided axes.
#    """
#    # Convert input lists to numpy arrays
#    pre_x_y = np.array(list_of_x_y_responses_pre)
#    x_y_responses = np.array(list_of_x_y_responses)
#    
#    # Split the arrays into x (input) and y (response) parts
#    pre_arr1, pre_arr2 = np.split(pre_x_y, 2, axis=1)
#    arr1, arr2 = np.split(x_y_responses, 2, axis=1)
#
#    # Flatten the arrays
#    pre_arr1 = np.ravel(pre_arr1)
#    pre_arr2 = np.ravel(pre_arr2)
#    arr1 = np.ravel(arr1)
#    arr2 = np.ravel(arr2)
#
#    # Perform curve fitting only for the 'gam' parameter
#    param_pre, _ = scipy.optimize.curve_fit(gama_fit, pre_arr1, pre_arr2, bounds=(0, 60))
#    param, _ = scipy.optimize.curve_fit(gama_fit, arr1, arr2, bounds=(0, 60))
#
#    # Generate x-values for plotting
#    pre_x = np.linspace(-0.5, 10, len(pre_arr2))
#    x = np.linspace(-0.5, 10, len(arr2))
#
#    # Generate fitted y-values using the optimized 'gam' parameter
#    pre_y = gama_fit(pre_x, param_pre[0])
#    y = gama_fit(x, param[0])
#
#    # Choose color based on cell type
#    color = bpf.CB_color_cycle[0] if cell_type == "learners" else bpf.CB_color_cycle[1]
#
#    # Plot the fitted curves
#    axs.plot(pre_x, pre_y, color='k', linestyle='-', alpha=0.8, label="pre_training", linewidth=3)
#    axs.plot(x, y, color=color, linestyle='-', alpha=0.8, label="post_training", linewidth=3)
#
#    # Set plot properties
#    axs.set_aspect(0.6)
#    axs.text(0.5, 0.9, f'γ_post = {np.around(param[0], 1)}', transform=axs.transAxes, fontsize=12, ha='center')
#    axs.text(0.5, 0.7, f'γ_pre = {np.around(param_pre[0], 1)}', transform=axs.transAxes, fontsize=12, ha='center')
#    
#    return x, y



def plot_expected_vs_observed_all_trials(alltrial_Df, mean_all_cell_df,
                                         sc_data_dict, cell_type, fig, axs1,
                                         axs2, axs3,axs4):
    """
    Plots the expected vs observed responses for all trials, separated by learners and non-learners.
    """
    # Filter cells based on type
    if cell_type == "learners":
        lrn = sc_data_dict["ap_cells"]["cell_ID"].unique()
        alltrial_Df = alltrial_Df[alltrial_Df["cell_ID"].isin(lrn)]
    elif cell_type == "non-learners":
        nlrn = sc_data_dict["an_cells"]["cell_ID"].unique()
        alltrial_Df = alltrial_Df[alltrial_Df["cell_ID"].isin(nlrn)]

    cell_grp = alltrial_Df.groupby(by='cell_ID')
    axs = [axs1, axs2, axs3]

    # Initialize lists to store responses
    response_dict = {i: ([], []) for i in range(3)}

    # Iterate over each cell
    for c, cell in cell_grp:
        if c == "2022_12_12_cell_5":
            continue

        pp_grps = cell.groupby(by="pre_post_status")
        for pp, pp_data in pp_grps:
            if pp not in ['pre', 'post_3']:
                continue

            color = bpf.pre_color if pp == "pre" else bpf.CB_color_cycle[0] if cell_type == "learners" else bpf.CB_color_cycle[1]

            pats = pp_data[pp_data["frame_status"] == "pattern"]["frame_id"].unique()
            trial_grp = pp_data.groupby(by="trial_no")

            for trial, trial_data in trial_grp:
                ppresp = pp_data[pp_data["trial_no"] == trial].reset_index(drop=True)
                for pat in pats:
                    pat_num = int(pat.split("_")[-1])
                    if pat_num not in response_dict:
                        continue

                    point_list = bpf.map_points_to_patterns(pat)
                    matching_values = ppresp.loc[ppresp["frame_id"] == pat, "max_trace"]
                    if len(matching_values) != 1:
                        raise ValueError(f"Expected exactly one matching row for frame_id={pat}, but found {len(matching_values)}.")
                    pat_val = float(matching_values.iloc[0])

                    #pat_val = float(ppresp[ppresp["frame_id"] == pat]["max_trace"].values)
                    point_sum_val = np.sum(ppresp[ppresp["frame_id"].isin(point_list)]["max_trace"])

                    # Append to the correct list based on 'pre' or 'post' status
                    response_list = response_dict[pat_num][0 if pp == "pre" else 1]
                    response_list.append([point_sum_val, pat_val])

                    # Plot scatter points
                    axs[pat_num].axline([0, 0], [1, 1], linestyle=':', color=bpf.CB_color_cycle[6], linewidth=2)
                    axs[pat_num].scatter(point_sum_val, pat_val, color=color, alpha=0.8, linewidth=1, marker=".")
                    axs[pat_num].spines[['right', 'top']].set_visible(False)
                    axs[pat_num].set_xlim(-0.5, 12)
                    axs[pat_num].set_ylim(-0.5, 12)
                    axs[pat_num].set_xticks(np.arange(-0.5, 12, 4))
                    axs[pat_num].set_yticks(np.arange(-0.5, 12, 4))
                    if pat_num == 0:
                        axs[pat_num].set_ylabel("observed\nresponse (mV)", fontsize=14)
                    else:
                        axs[pat_num].set_yticklabels([])

                    if cell_type == "learners":
                        axs[pat_num].set_xticklabels([])
                    else:
                        if pat_num == 1:
                            axs[pat_num].set_xlabel("expected response (mV)")

    # Fit and plot the gamma model for each pattern
    for pat_num in range(3):
        pre_responses, post_responses = response_dict[pat_num]
        eq_fit(pre_responses, post_responses, pat_num, cell_type, fig, axs[pat_num])
    # Fit and plot the gamma model for pos_3 trained v/s non-overlap
    pre_trained, post_trained = response_dict[0]
    pre_nonoverlap, post_nonoverlap = response_dict[2]
    eq_fit(post_trained, post_nonoverlap, pat_num, cell_type, fig, axs4)



    # Add a legend for non-learners in pattern 1
    if cell_type == "non-learners":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=bpf.pre_color, 
                   markersize=10, label='Pre training'),
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=bpf.CB_color_cycle[0],
                   markersize=10, label='Learner\npost training'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=bpf.CB_color_cycle[1],
                   markersize=10, label='Non-learner\npost training')
        ]
        axs[1].legend(handles=legend_elements, 
                      bbox_to_anchor=(0.5,-0.6),ncol=3,
                      loc='center', frameon=False)


#def plot_expected_vs_observed_all_trials(alltrial_Df, mean_all_cell_df, sc_data_dict, cell_type, fig, axs1, axs2, axs3):
#    """
#    Plots the expected vs observed responses for all trials, separated by learners and non-learners.
#    """
#    # Filter cells based on type
#    if cell_type == "learners":
#        lrn = sc_data_dict["ap_cells"]["cell_ID"].unique()
#        alltrial_Df = alltrial_Df[alltrial_Df["cell_ID"].isin(lrn)]
#    elif cell_type == "non-learners":
#        nlrn = sc_data_dict["an_cells"]["cell_ID"].unique()
#        alltrial_Df = alltrial_Df[alltrial_Df["cell_ID"].isin(nlrn)]
#
#    cell_grp = alltrial_Df.groupby(by='cell_ID')
#    axs = [axs1, axs2, axs3]
#
#    # Initialize lists to store responses
#    response_dict = {i: ([], []) for i in range(3)}
#
#    # Iterate over each cell
#    for c, cell in cell_grp:
#        if c == "2022_12_12_cell_5":
#            continue
#
#        pp_grps = cell.groupby(by="pre_post_status")
#        for pp, pp_data in pp_grps:
#            if pp not in ['pre', 'post_3']:
#                continue
#
#            color = bpf.pre_color if pp == "pre" else bpf.CB_color_cycle[0] if cell_type == "learners" else bpf.CB_color_cycle[1]
#
#            pats = pp_data[pp_data["frame_status"] == "pattern"]["frame_id"].unique()
#            trial_grp = pp_data.groupby(by="trial_no")
#
#            for trial, trial_data in trial_grp:
#                ppresp = pp_data[pp_data["trial_no"] == trial].reset_index(drop=True)
#                for pat in pats:
#                    pat_num = int(pat.split("_")[-1])
#                    if pat_num not in response_dict:
#                        continue
#
#                    point_list = bpf.map_points_to_patterns(pat)
#                    pat_val = float(ppresp[ppresp["frame_id"] == pat]["max_trace"].values)
#                    point_sum_val = np.sum(ppresp[ppresp["frame_id"].isin(point_list)]["max_trace"])
#
#                    # Append to the correct list based on 'pre' or 'post' status
#                    response_list = response_dict[pat_num][0 if pp == "pre" else 1]
#                    response_list.append([point_sum_val, pat_val])
#
#                    # Plot scatter points
#                    axs[pat_num].axline([0, 0], [1, 1], linestyle=':', color=bpf.CB_color_cycle[6], linewidth=2)
#                    axs[pat_num].scatter(point_sum_val, pat_val, color=color, alpha=0.8, linewidth=1, marker=".")
#                    axs[pat_num].spines[['right', 'top']].set_visible(False)
#                    axs[pat_num].set_xlim(-0.5, 12)
#                    axs[pat_num].set_ylim(-0.5, 12)
#                    axs[pat_num].set_xticks(np.arange(-0.5, 12, 4))
#                    axs[pat_num].set_yticks(np.arange(-0.5, 12, 4))
#                    
#                    if pat_num == 0:
#                        axs[pat_num].set_ylabel("observed\nresponse (mV)", fontsize=14)
#                    else:
#                        axs[pat_num].set_yticklabels([])
#
#                    if cell_type == "learners":
#                        axs[pat_num].set_xticklabels([])
#                    else:
#                        if pat_num == 1:
#                            axs[pat_num].set_xlabel("expected response (mV)")
#
#    # Fit and plot the gamma model for each pattern
#    for pat_num in range(3):
#        pre_responses, post_responses = response_dict[pat_num]
#        eq_fit(pre_responses, post_responses, pat_num, cell_type, fig, axs[pat_num])



#def gama_fit(expt,alp,bet,gam):
#    bet=1
#    alp=0
#    return expt-(((bet*expt)/(gam+expt))*expt)-alp
#
#def eq_fit(list_of_x_y_responses_pre,list_of_x_y_responses,pat_num,
#           cell_type,fig, axs):
#    x_y_responses = np.array(list_of_x_y_responses)
#    pre_x_y = np.array(list_of_x_y_responses_pre)
#    pre_arr1,pre_arr2=np.split(pre_x_y,2,axis=1)
#    pre_arr1 =np.ravel(pre_arr1)
#    pre_arr2 =np.ravel(pre_arr2)
#    arr1,arr2=np.split(x_y_responses,2,axis=1)
#    arr1 = np.ravel(arr1)
#    arr2 = np.ravel(arr2)
#    print(f"array shape: {np.shape(pre_arr2)}")
#
#    param_pre, _param_pre = scipy.optimize.curve_fit(gama_fit,pre_arr1,pre_arr2, bounds=(0,60))
#    param, _param = scipy.optimize.curve_fit(gama_fit,arr1,arr2, bounds=(0,60))
#
#    #pre_x = np.arange(int(np.floor(np.min(pre_arr1))),int(np.ceil(np.max(pre_arr2))),0.5)
#    pre_x  = np.linspace(-0.5, 10,len(pre_arr2))
#    #pre_x  = np.linspace(0, np.max(pre_arr2)+3,len(pre_arr2))
#    #x = np.arange(int(np.floor(np.min(arr1))),int(np.ceil(np.max(arr1))),0.5)
#    #x  = np.linspace(0, np.max(arr2)+3,len(arr2))
#    x  = np.linspace(-0.5, 10,len(arr2))
#    pre_y = gama_fit(pre_x,param_pre[0],param_pre[1],param_pre[2])
#    y = gama_fit(x,param[0],param[1],param[2])
#    if cell_type=="learners":
#        color=bpf.CB_color_cycle[0]
#    else:
#        color= bpf.CB_color_cycle[1]
#    axs.plot(pre_x, pre_y, color='k', linestyle='-', alpha=0.8, label="pre_training",linewidth=3)
#    axs.plot(x, y, color=color, linestyle='-', alpha=0.8, label="post_training",linewidth=3)
#    #axs[pat_num].text(1,10, f"r ={round(r_value*r_value,2)}", fontsize = 10)
#    axs.set_aspect(0.6)
#    axs.text(0.5,0.9,f'γ_post = {np.around(param[-1],1)}',transform=axs.transAxes,    
#             fontsize=12, ha='center', va='center')
#    axs.text(0.5,0.7,f'γ_pre = {np.around(param_pre[-1],1)}',transform=axs.transAxes,
#             fontsize=12, ha='center', va='center')
#    
#    return x, y
#
#def plot_expected_vs_observed_all_trials(alltrial_Df,
#                                         mean_all_cell_df,
#                                         sc_data_dict,
#                                         cell_type,
#                                         fig,axs1,axs2,axs3):
#    if cell_type=="learners":
#        lrn=sc_data_dict["ap_cells"]["cell_ID"].unique()
#        alltrial_Df = alltrial_Df[alltrial_Df["cell_ID"].isin(lrn)]
#    elif cell_type=="non-learners":
#        nlrn=sc_data_dict["an_cells"]["cell_ID"].unique()
#        alltrial_Df=alltrial_Df[alltrial_Df["cell_ID"].isin(nlrn)]
#    cell_grp  = alltrial_Df.groupby(by='cell_ID')
#    axs=[axs1,axs2,axs3]
#    all_resp_pat_0 =[]
#    all_resp_pat_1 =[]
#    all_resp_pat_2 =[]
#    pre_all_resp_pat_0 =[]
#    pre_all_resp_pat_1 =[]
#    pre_all_resp_pat_2 =[]
#    
#    for c, cell in cell_grp:
#        if c=="2022_12_12_cell_5":
#            continue
#        pp_grps = cell.groupby(by="pre_post_status")
#        for pp, pp_data in pp_grps:
#            if pp not in ['pre','post_3']:
#                continue
#            else:
#                pp =pp
#                #print(f"found pp stat")
#            if pp=="pre":
#                color=bpf.pre_color
#            else:
#                if cell_type=="learners":
#                    color = bpf.CB_color_cycle[0]
#                else:
#                    color = bpf.CB_color_cycle[1]
#            pats= pp_data[pp_data["frame_status"]=="pattern"]["frame_id"].unique()
#            trial_grp  = pp_data.groupby(by="trial_no")
#            for trial, trial_data in trial_grp:
#                ppresp =pp_data.copy()
#                ppresp = ppresp[ppresp["trial_no"]==trial]
#                ppresp.reset_index(drop=True)
#                for pat in pats:
#                    pat_num = int(pat.split("_")[-1])
#                    point_list = bpf.map_points_to_patterns(pat)
#                    pat_val=float(ppresp[ppresp["frame_id"]==pat]["max_trace"].values)
#                    point_sum_val =np.sum(np.array(ppresp[ppresp["frame_id"].isin(point_list)]["max_trace"]))
#                    pat_val_nrm = pat_val#-pat_val
#                    point_sum_val_nrm = point_sum_val#-pat_val
#                    #print(f"pat, point: {pat_val_nrm},{point_sum_val}")
#                    if pp!="pre":
#                        if pat_num==0:
#                            all_resp_pat_0.append([point_sum_val_nrm,pat_val_nrm])
#                        elif pat_num==1:
#                            all_resp_pat_1.append([point_sum_val_nrm,pat_val_nrm])
#                        elif pat_num==2:
#                            all_resp_pat_2.append([point_sum_val_nrm,pat_val_nrm])
#                        else:
#                            continue
#                    elif pp=="pre":
#                        if pat_num==0:
#                            pre_all_resp_pat_0.append([point_sum_val_nrm,pat_val_nrm])
#                        elif pat_num==1:
#                            pre_all_resp_pat_1.append([point_sum_val_nrm,pat_val_nrm])
#                        elif pat_num==2:
#                            pre_all_resp_pat_2.append([point_sum_val_nrm,pat_val_nrm])
#                        else:
#                            continue
#                    axs[pat_num].axline([0,0], [1,1], linestyle=':',
#                                        color=bpf.CB_color_cycle[6], label="linear sum",linewidth=2)
#                    axs[pat_num].scatter(point_sum_val_nrm,pat_val_nrm,
#                                         color=color, label=pp, alpha=0.8,
#                                         linewidth=1,marker=".")
#                    axs[pat_num].spines[['right', 'top']].set_visible(False)
#                    axs[pat_num].set_xlim(-0.5,12)
#                    axs[pat_num].set_ylim(-0.5,12)
#                    axs[pat_num].set_xticks(np.arange(-0.5,12,4))
#                    axs[pat_num].set_yticks(np.arange(-0.5,12,4))
#                    
#                    if pat_num==0:
#                        axs[pat_num].set_ylabel("observed\nresponse (mV)", fontsize=14)
#                        
#                    else:
#                        axs[pat_num].set_yticklabels([])
#                    if cell_type=="learners":
#                        axs[pat_num].set_title(None)
#                        axs[pat_num].set_xlabel(None)
#                        axs[pat_num].set_xticklabels([])
#                    else:
#                        if pat_num==1:
#                            axs[pat_num].set_xlabel("expected response (mV)")
#                        else:
#                            axs[pat_num].set_title(None)
#                    if pp!="pre":
#                        if pat_num==0:
#                            all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        elif pat_num==1:
#                            all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        elif pat_num==2:
#                            all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        else:
#                            continue
#                    elif pp=="pre":
#                        if pat_num==0:
#                            pre_all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        elif pat_num==1:
#                            pre_all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        elif pat_num==2:
#                            pre_all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
#                        else:
#                            continue
#    eq_fit(pre_all_resp_pat_0,all_resp_pat_0,0,cell_type, fig, axs1)
#    eq_fit(pre_all_resp_pat_1,all_resp_pat_1,1,cell_type, fig, axs2)
#    eq_fit(pre_all_resp_pat_2,all_resp_pat_2,2,cell_type, fig, axs3)

#def plot_expected_vs_observed(pd_cell_data_mean_cell_grp,cell_type,f_norm_status,fig,axs1,axs2,axs3):
#    pd_cell_data_mean_cell_grp = pd_cell_data_mean_cell_grp[pd_cell_data_mean_cell_grp["pre_post_status"]!="post_5"]
#    pd_cell_data_mean_cell_grp = pd_cell_data_mean_cell_grp.groupby(by='cell_ID')
#    axs=[axs1,axs2,axs3]
#    
#    all_resp_pat_0 =[]
#    all_resp_pat_1 =[]
#    all_resp_pat_2 =[]
#    pre_all_resp_pat_0 =[]
#    pre_all_resp_pat_1 =[]
#    pre_all_resp_pat_2 =[]
#    
#    for c, cell in pd_cell_data_mean_cell_grp:
#        pp_grps = cell.groupby(by="pre_post_status")
#        cl = len(cell["pre_post_status"].unique())-2
#        for pp, ppresp in pp_grps:
#            if pp not in ['pre','post_3']:
#                continue
#            else:
#                pp =pp
#                #print(f"found pp stat")
#                
#            if pp=="pre":
#                color=bpf.pre_color
#            #elif pp=="post_4":
#            #    cmx= int(pp.split("_")[-1])/cl
#            #    color=colorFader(post_color,post_late,mix=cmx)
#            else:
#                #continue
#                cmx= int(pp.split("_")[-1])/cl
#                color=bpf.colorFader(bpf.post_color,bpf.post_late,mix=cmx)
#                if cell_type=="learners":
#                    color = bpf.CB_color_cycle[0]
#                else:
#                    color = bpf.CB_color_cycle[1]
#            pats= ppresp[ppresp["frame_status"]=="pattern"]["frame_id"].unique()
#            for pat in pats:
#                pat_num = int(pat.split("_")[-1])
#                point_list = bpf.map_points_to_patterns(pat)
#                pat_val = float(ppresp[ppresp["frame_id"]==pat]["max_trace"])
#                point_sum_val = float(np.sum(np.array((ppresp[ppresp["frame_id"].isin(point_list)]["max_trace"]))))
#                pat_val_nrm = pat_val#-pat_val
#                point_sum_val_nrm = point_sum_val#-pat_val
#                #print(f"pat, point: {pat_val_nrm},{point_sum_val}")
#                axs[pat_num].axline([0,0], [1,1], linestyle=':',
#                                    color=bpf.CB_color_cycle[6], label="linear sum",linewidth=2)
#                axs[pat_num].scatter(point_sum_val_nrm,pat_val_nrm,color=color, label=pp, alpha=0.8,linewidth=2)
#                axs[pat_num].spines[['right', 'top']].set_visible(False)
#                axs[pat_num].set_xlim(-1,12)
#                axs[pat_num].set_ylim(-1,12)
#                axs[pat_num].set_xticks(np.arange(-1,12,4))
#                axs[pat_num].set_yticks(np.arange(-1,12,4))
#                
#                if pat_num==0:
#                    axs[pat_num].set_ylabel("observed\nresponse (mV)", fontsize=14)
#                    
#                else:
#                    axs[pat_num].set_yticklabels([])
#                if cell_type=="learners":
#                    axs[pat_num].set_title(None)
#                    axs[pat_num].set_xlabel(None)
#                    axs[pat_num].set_xticklabels([])
#                else:
#                    if pat_num==1:
#                        axs[pat_num].set_xlabel("expected response (mV)")
#                    else:
#                        axs[pat_num].set_title(None)
#                if pp!="pre":
#                    if pat_num==0:
#                        all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    elif pat_num==1:
#                        all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    elif pat_num==2:
#                        all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    else:
#                        continue
#                elif pp=="pre":
#                    if pat_num==0:
#                        pre_all_resp_pat_0.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    elif pat_num==1:
#                        pre_all_resp_pat_1.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    elif pat_num==2:
#                        pre_all_resp_pat_2.append([point_sum_val_nrm,float(pat_val_nrm)])
#                    else:
#                        continue
#                    
#
#    #poly_fit_sums(pre_all_resp_pat_0,all_resp_pat_0,0,fig, axs)
#    #poly_fit_sums(pre_all_resp_pat_1,all_resp_pat_1,1,fig, axs)
#    #poly_fit_sums(pre_all_resp_pat_2,all_resp_pat_2,2,fig, axs)
#    eq_pre_max = np.max(pre_all_resp_pat_0)
#    eq_post_max = np.max(all_resp_pat_0)
#    eq_fit(pre_all_resp_pat_0,all_resp_pat_0,0,eq_pre_max,eq_post_max,cell_type, fig, axs1)
#    eq_fit(pre_all_resp_pat_1,all_resp_pat_1,1,eq_pre_max,eq_post_max,cell_type, fig, axs2)
#    eq_fit(pre_all_resp_pat_2,all_resp_pat_2,2,eq_pre_max,eq_post_max,cell_type, fig, axs3)

#def plot_initial_final_wt(feature_extracted_data,sc_data_dict,fig,axs):
#    learners=sc_data_dict["ap_cells"]["cell_ID"].unique()
#    non_learners=sc_data_dict["an_cells"]["cell_ID"].unique()
#    deselect_list = ["no_frame","inR"]
#    feature_extracted_data=feature_extracted_data[~feature_extracted_data["frame_status"].isin(deselect_list)]
#    cell_grp = feature_extracted_data.groupby(by="cell_ID")
#    for cell, cell_data in cell_grp:
#        if cell in learners:
#            color=bpf.CB_color_cycle[0]
#        elif cell in non_learners:
#            color=bpf.CB_color_cycle[1]
#        else:
#            continue
#        frame_grp = cell_data.groupby(by="frame_status")
#        for frame, frame_data in frame_grp:
#            if "point" not in frame:
#                continue
#            else:
#                point_num = int(frame.split("_")[-1])
#                pps_grp=frame_data.groupby(by="pre_post_status")
#                for pps,pps_data in pps_grp:
#                    axs.scatter(point_num,pps_data["max_trace"])
#


def plot_figure_7(extracted_feature_pickle_file_path,
                  sum_illustration_path,
                  cell_categorised_pickle_file,
                  cell_stats_pickle_file,
                  all_trials_path,
                  outdir,learner_cell=learner_cell,
                  non_learner_cell=non_learner_cell):
    deselect_list = ["no_frame","inR","point"]
    feature_extracted_data = pd.read_pickle(extracted_feature_pickle_file_path)
    cell_stats_df = pd.read_hdf(cell_stats_pickle_file)
    print(f"cell stat df : {cell_stats_df}")
    alltrial_Df=pd.read_pickle(all_trials_path)
    single_cell_df = feature_extracted_data.copy()
    learner_cell_df = single_cell_df.copy()
    non_learner_cell_df = single_cell_df.copy()
    learner_cell_df = single_cell_df[(single_cell_df["cell_ID"]==learner_cell)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    non_learner_cell_df = single_cell_df[(single_cell_df["cell_ID"]==non_learner_cell)&(single_cell_df["pre_post_status"].isin(selected_time_points))]
    sc_data_dict = pd.read_pickle(cell_categorised_pickle_file)
    sc_data_df = pd.concat([sc_data_dict["ap_cells"],
                            sc_data_dict["an_cells"]]).reset_index(drop=True)
    print(f"sc data : {sc_data_df['cell_ID'].unique()}")

    sum_illust= pillow.Image.open(sum_illustration_path)
    # Check if the image has an alpha channel (transparency)
    if sum_illust.mode == 'RGBA':
        # Create a white background using the correct Image class method
        white_bg = pillow.Image.new("RGB", sum_illust.size, (255, 255, 255))
        # Paste the image on top of the white background, handling transparency
        white_bg.paste(sum_illust, mask=sum_illust.split()[3])  # 3 is the alpha channel
        sum_illust = white_bg

    # If the image is not RGB, convert it to RGB
    if sum_illust.mode != "RGB":
        sum_illust = sum_illust.convert("RGB")
    
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



    #plot sumamtion illustration
    axs_illu = fig.add_subplot(gs[0:3,1:5])
    plot_image(sum_illust,axs_illu,-0.125,0,2)
    axs_illu.text(0.05,0.925,'A',transform=axs_illu.transAxes,
                 fontsize=16, fontweight='bold', ha='center', va='center')
    #plot patterns
    axs_pat_1 = fig.add_subplot(gs[3:4,0:1])
    axs_pat_2 = fig.add_subplot(gs[3:4,2:3])
    axs_pat_3 = fig.add_subplot(gs[3:4,4:5])

    plot_patterns(axs_pat_1,axs_pat_2,axs_pat_3,0.05,0,1)



    #plot summation for learners and leaners
    axs_ex_sm1 = fig.add_subplot(gs[4:6,0:2])
    axs_ex_sm2 = fig.add_subplot(gs[4:6,2:4])
    axs_ex_sm3 = fig.add_subplot(gs[4:6,4:6])
    axs_ex_sm3_ = fig.add_subplot(gs[9:11,0:2])
    plot_expected_vs_observed_all_trials(alltrial_Df,
                                         feature_extracted_data,
                                         sc_data_dict,"learners",
                              fig,axs_ex_sm1,axs_ex_sm2,axs_ex_sm3,axs_ex_sm3_)
    axs_ex_sm_l_list = [axs_ex_sm1,axs_ex_sm2,axs_ex_sm3]
    label_axis(axs_ex_sm_l_list, "B", xpos=-0.1, ypos=1.08)
    #axs_ex_sm2.set_title("learners")
    axs_ex_sm2.set_xlabel(None)
    axs_ex_sm4 = fig.add_subplot(gs[6:8,0:2])
    axs_ex_sm5 = fig.add_subplot(gs[6:8,2:4])
    axs_ex_sm6 = fig.add_subplot(gs[6:8,4:6])
    axs_ex_sm6_ = fig.add_subplot(gs[9:11,2:4])
    plot_expected_vs_observed_all_trials(alltrial_Df,
                                         feature_extracted_data,
                                         sc_data_dict,"non-learners",
                              fig,axs_ex_sm4,axs_ex_sm5,axs_ex_sm6,axs_ex_sm6_)
    #axs_ex_sm5.set_title("non-learners")
    
    axs_ex_sm_nl_list= [axs_ex_sm4,axs_ex_sm5,axs_ex_sm6]
    move_axis(axs_ex_sm_nl_list,xoffset=0,yoffset=-0.01,pltscale=1) 
    label_axis(axs_ex_sm_nl_list,"C", xpos=-0.1, ypos=1.08)



    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor =(0.5, 0.175),
    #           ncol = 6,title="Legend",
    #           loc='upper center')#,frameon=False)#,loc='lower center'    
    #

    plt.tight_layout()
    outpath = f"{outdir}/figure_7_with_stats.png"
    #outpath = f"{outdir}/figure_7.svg"
    #outpath = f"{outdir}/figure_7_with_stats.pdf"
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
    parser.add_argument('--alltrials-path', '-t'
                        , required = False,default ='./', type=str
                        , help = 'path to pickle file with all trials'
                        'all cells data in pickle'
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
    sum_illustration_path = Path(args.illustration_path)
    cell_stat_path = Path(args.cellstat_path)
    all_trials_path= Path(args.alltrials_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'Figure_7'
    globoutdir.mkdir(exist_ok=True, parents=True)
    print(f"pkl path : {pklpath}")
    plot_figure_7(pklpath,sum_illustration_path,
                  scpath,cell_stat_path,all_trials_path,globoutdir)
    #print(f"illustration path: {illustration_path}")




if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
