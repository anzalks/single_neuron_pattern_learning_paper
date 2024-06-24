__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"


import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


pre_color = "000000" #pre_color black 
post_color = "#377eb8" #post_color blue 
post_late = "#de6f00" # post late orange

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (atmix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        print(f"pvalue:{pvalue}")
        #return str(pvalue)
        return f"**** p= {np.around(pvalue,3)}"
    elif pvalue <= 0.001:
        return f"*** p= {np.around(pvalue,3)}"
    elif pvalue <= 0.01:
        return f"** p= {np.around(pvalue,3)}"
    elif pvalue <= 0.05:
        #return str(pvalue)
        return f"* p= {np.around(pvalue,3)}"
    else:
        return "ns"

def set_plot_properties():
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'arial'#arial prefered
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 11
"""    
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# Example usage of color cycler for different plot types
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bar_colors = ['#d62728', '#9467bd', '#8c564b']
scatter_colors = ['#e377c2', '#7f7f7f', '#bcbd22']
"""
    
def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def substract_baseline(trace,sampling_rate=20000, bl_period_in_ms=5):
    bl_period = bl_period_in_ms/1000
    bl_duration = int(sampling_rate*bl_period)
    bl = np.mean(trace[:bl_duration])
    bl_trace = trace-bl
    return bl_trace

def map_points_to_patterns(pattern):
    if pattern=='pattern_0':
        points=['point_0', 'point_1', 'point_2', 'point_3', 'point_4']
    elif pattern =='pattern_1':
        points=['point_2', 'point_3', 'point_4','point_5', 'point_6']
    elif pattern=='pattern_2':
        points= ['point_7', 'point_8', 'point_9','point_10', 'point_11']
    else:
        points=None
        print("pat_num out of bound")
    return points

def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None



def create_grid_image(first_spot_grid_point, spot_proportional_size, image_size=(1024, 480), grid_size=(24, 24), num_spots=5, spot_color=(0, 255, 255)):
    """
    Creates a custom grid image with bright spots starting from a specified grid point.

    Parameters:
    - first_spot_grid_point (int): The grid point (0-23) for the first bright spot on the x-axis.
    - spot_proportional_size (int): The proportional size of the bright spots (number of grid cells).
    - image_size (tuple): The size of the image (width, height).
    - grid_size (tuple): The size of the grid (columns, rows).
    - num_spots (int): The number of bright spots.
    - spot_color (tuple): The color of the bright spots (R, G, B).

    Returns:
    - PIL.Image: The generated image.
    """
    # Create the image with a black background
    image = Image.new("RGB", image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Calculate grid cell size
    grid_cell_width = image_size[0] // grid_size[0]
    grid_cell_height = image_size[1] // grid_size[1]

    # Calculate the size of the bright spots based on the proportional size
    spot_width = grid_cell_width * spot_proportional_size
    spot_height = grid_cell_height * spot_proportional_size

    # Calculate the maximum possible starting grid point to keep the spots inside the image
    max_starting_grid_point = grid_size[0] - (num_spots * spot_proportional_size + (num_spots - 1) * spot_proportional_size // 2)
    first_spot_grid_point = min(first_spot_grid_point, max_starting_grid_point)

    # Calculate the starting position for the first bright spot on the x-axis
    first_spot_x = first_spot_grid_point * grid_cell_width
    y = (image_size[1] - spot_height) // 2

    # Define the gap size between the bright spots
    gap_size = spot_width // 2

    # Add bright spots starting from the defined grid point, arranged linearly with one spot gap in between
    for i in range(num_spots):
        x = first_spot_x + i * (spot_width + gap_size)
        draw.rectangle([x, y, x + spot_width, y + spot_height], fill=spot_color)

    return image
