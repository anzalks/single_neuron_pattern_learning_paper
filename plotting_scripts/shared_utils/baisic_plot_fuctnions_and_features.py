__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"


import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import PIL as pillow
from PIL import Image, ImageDraw, ImageFont 
import os

# Global subplot label control
# Check environment variable first, then default to True
_env_labels = os.environ.get('SUBPLOT_LABELS_ENABLED', 'True')
SUBPLOT_LABELS_ENABLED = _env_labels.lower() in ('true', '1', 'yes', 'on')

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
        #print(f"pvalue:{pvalue}")
        #return str(pvalue)
        #pvalue=str(np.around(pvalue,5))
        return f"****"
        #return f"**** \np= {pvalue}"
    elif pvalue <= 0.001:
        #return str(pvalue)
        return f"***"
        #return str(pvalue)
        #return f"*** \np= {np.around(pvalue,4)}"
    elif pvalue <= 0.01:
        #return str(pvalue)
        #return f"** \np= {np.around(pvalue,3)}"
        return f"**"
    elif pvalue <= 0.05:
        #return str(pvalue)
        #return f"* p= {np.around(pvalue,3)}"
        return f"*"
    else:
        #return f"ns p= {np.around(pvalue,3)}"
        return "ns"

def set_plot_properties():
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.family'] = 'arial'#arial prefered
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 11
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['text.usetex'] = False
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



#def create_grid_image(first_spot_grid_point, spot_proportional_size=1.5, image_size=(1024, 480), grid_size=(24, 24), num_spots=5, spot_color=(0, 255, 255)):
#    """
#    Creates a custom grid image with bright spots starting from a specified grid point.
#
#    Parameters:
#    - first_spot_grid_point (int): The grid point (0-23) for the first bright spot on the x-axis.
#    - spot_proportional_size (int): The proportional size of the bright spots (number of grid cells).
#    - image_size (tuple): The size of the image (width, height).
#    - grid_size (tuple): The size of the grid (columns, rows).
#    - num_spots (int): The number of bright spots.
#    - spot_color (tuple): The color of the bright spots (R, G, B).
#
#    Returns:
#    - PIL.Image: The generated image.
#    """
#    # Create the image with a black background
#    image = Image.new("RGB", image_size, (0, 0, 0))
#    draw = ImageDraw.Draw(image)
#
#    # Calculate grid cell size
#    grid_cell_width = image_size[0] // grid_size[0]
#    grid_cell_height = image_size[1] // grid_size[1]
#
#    # Calculate the size of the bright spots based on the proportional size
#    spot_width = grid_cell_width * spot_proportional_size
#    spot_height = grid_cell_height * spot_proportional_size
#
#    # Calculate the maximum possible starting grid point to keep the spots inside the image
#    max_starting_grid_point = grid_size[0] - (num_spots * spot_proportional_size + (num_spots - 1) * spot_proportional_size // 2)
#    first_spot_grid_point = min(first_spot_grid_point, max_starting_grid_point)
#
#    # Calculate the starting position for the first bright spot on the x-axis
#    first_spot_x = first_spot_grid_point * grid_cell_width
#    y = (image_size[1] - spot_height) // 2
#
#    # Define the gap size between the bright spots
#    gap_size = spot_width // 2
#
#    # Add bright spots starting from the defined grid point, arranged linearly with one spot gap in between
#    for i in range(num_spots):
#        x = first_spot_x + i * (spot_width + gap_size)
#        draw.rectangle([x, y, x + spot_width, y + spot_height], fill=spot_color)
#
#    return image



def create_grid_image(first_spot_grid_point, spot_proportional_size=1.5,
                      image_size=(1024, 480), grid_size=(24, 24), num_spots=5,
                      spot_color=(0, 0, 0), background_color=(255, 255, 255),
                      border=True):
    """
    Creates a custom grid image with bright spots starting from a specified grid point,
    with an optional black border around the entire image.

    Parameters:
    - first_spot_grid_point (int): The grid point (0-23) for the first bright spot on the x-axis.
    - spot_proportional_size (int): The proportional size of the bright spots (number of grid cells).
    - image_size (tuple): The size of the image (width, height).
    - grid_size (tuple): The size of the grid (columns, rows).
    - num_spots (int): The number of bright spots.
    - spot_color (tuple): The color of the bright spots (R, G, B).
    - background_color (tuple): The background color of the image (R, G, B).
    - border (bool): Whether to add a black border around the entire image.

    Returns:
    - PIL.Image: The generated image.
    """
    # Add border thickness if requested
    border_thickness = 5 if border else 0
    bordered_image_size = (image_size[0] + 2 * border_thickness, image_size[1] + 2 * border_thickness)

    # Create the image with a specified background color and border (if any)
    image = Image.new("RGB", bordered_image_size, (0, 0, 0) if border else background_color)
    draw = ImageDraw.Draw(image)

    # Inner area where the grid will be drawn
    inner_image = Image.new("RGB", image_size, background_color)
    inner_draw = ImageDraw.Draw(inner_image)

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
        inner_draw.rectangle([x, y, x + spot_width, y + spot_height], fill=spot_color)

    # Paste the inner image onto the main image with the border
    image.paste(inner_image, (border_thickness, border_thickness))

    return image

#def create_grid_points_with_text(first_spot_grid_points, spot_proportional_size=0.5, image_size=(300, 100), grid_size=(24, 24), spot_color=(0, 255, 255), padding=30, background_color=(255, 255, 255), text_color=(0, 0, 0), font_size=20, show_text=True, num_columns=3, txt_spacing=20, min_padding_above_text=10):
#    """
#    Creates a single image composed of multiple individual images arranged in multiple rows. Optionally, text is shown above each spot.
#
#    Parameters:
#    - first_spot_grid_points (list of int): A list of grid points for the bright spots on the x-axis for each individual image.
#    - spot_proportional_size (int): The proportional size of the bright spots (number of grid cells).
#    - image_size (tuple): The size of each individual image (width, height).
#    - grid_size (tuple): The size of the grid (columns, rows).
#    - spot_color (tuple): The color of the bright spots (R, G, B).
#    - padding (int): The padding (in pixels) to add between each image.
#    - background_color (tuple): The background color for the padding (default: white).
#    - text_color (tuple): The color of the text (default: black).
#    - font_size (int): Font size for the text label on each rectangle.
#    - show_text (bool): Whether or not to display the text above each spot.
#    - num_columns (int): Number of images to display in each row.
#    - txt_spacing (int): Additional space between the text and the image.
#    - min_padding_above_text (int): Minimum space to maintain between the top of the text and the top of the image.
#    
#    Returns:
#    - PIL.Image: The combined image with all individual images arranged in multiple rows.
#    """
#
#    num_images = len(first_spot_grid_points)
#    num_rows = (num_images + num_columns - 1) // num_columns  # Calculate how many rows are needed
#
#    # Calculate the total height of the image, including padding and space for text
#    text_height = font_size + txt_spacing + min_padding_above_text
#    combined_image_height = (image_size[1] + (text_height if show_text else 0)) * num_rows + padding * (num_rows + 1)
#    combined_image_width = image_size[0] * num_columns + padding * (num_columns + 1)
#
#    # Create the base combined image
#    combined_image = Image.new("RGB", (combined_image_width, combined_image_height), background_color)
#    
#    # Use a TrueType font if available
#    try:
#        # Specify the path to a TrueType font (use Arial as an example)
#        font_path = "/Library/Fonts/Arial.ttf"
#        font = pillow.ImageFont.truetype(font_path, font_size)
#    except IOError:
#        print("TrueType font not found. Using default bitmap font.")
#        font = pillow.ImageFont.load_default()
#
#    for idx, spot_grid_point in enumerate(first_spot_grid_points):
#        # Create each individual image with one cyan rectangle
#        image = Image.new("RGB", image_size, (0, 0, 0))
#        draw = ImageDraw.Draw(image)
#
#        # Calculate grid cell size
#        grid_cell_width = image_size[0] // grid_size[0]
#        grid_cell_height = image_size[1] // grid_size[1]
#
#        # Calculate the size of the bright spot based on the proportional size
#        spot_width = int(grid_cell_width * spot_proportional_size)
#        spot_height = int(grid_cell_height * spot_proportional_size)
#
#        # Ensure the starting grid point keeps the spot inside the image
#        spot_grid_point_adjusted = min(spot_grid_point, grid_size[0] - 1)
#
#        # Calculate the starting position for the bright spot on the x-axis
#        first_spot_x = spot_grid_point_adjusted * grid_cell_width
#        y = (image_size[1] - spot_height) // 2
#
#        # Add a single bright spot at the defined grid point
#        draw.rectangle([first_spot_x, y, first_spot_x + spot_width, y + spot_height], fill=spot_color)
#
#        # Calculate the x and y position for this image in the combined grid
#        col = idx % num_columns  # Current column
#        row = idx // num_columns  # Current row
#        
#        x_position = col * image_size[0] + padding * (col + 1)
#        y_position = row * (image_size[1] + text_height if show_text else 0) + padding * (row + 1)
#
#        # Paste the individual image into the combined image
#        combined_image.paste(image, (x_position, y_position))
#
#        # Conditionally add text if show_text is True
#        if show_text:
#            # Use index + 1 as the text label
#            text = f"{idx + 1}"  # This will use the index (0-based) + 1
#
#            # Use textbbox to get the bounding box of the text
#            draw_combined = ImageDraw.Draw(combined_image)
#            text_bbox = draw_combined.textbbox((0, 0), text, font=font)
#            text_width = text_bbox[2] - text_bbox[0]
#            text_height = text_bbox[3] - text_bbox[1]
#
#            # Calculate text size and position it centered in the padding area above the rectangle
#            text_x = x_position + (image_size[0] - text_width) // 2
#            text_y = max(min_padding_above_text, y_position - (padding // 2 + txt_spacing))  # Ensure text is within bounds
#
#            # Draw the text label, ensuring it doesn't go beyond the top of the image
#            draw_combined.text((text_x, text_y), text, fill=text_color, font=font)
#
#    return combined_image



def create_grid_points_with_text(
    first_spot_grid_points, 
    spot_proportional_size=0.5, 
    image_size=(300, 100), 
    grid_size=(24, 24), 
    spot_color=(0, 0, 0), 
    padding=30, 
    background_color=(255, 255, 255), 
    text_color=(0, 0, 0), 
    font_size=20, 
    show_text=True, 
    num_columns=3, 
    txt_spacing=20, 
    min_padding_above_text=10,
    image_background_color=(255, 255, 255),  # New parameter for individual image background
    border=True  # New parameter to add border
):
    """
    Creates a single image composed of multiple individual images arranged in multiple rows. Optionally, text is shown above each spot.
    Includes options for background color and border around each spot image.

    Parameters:
    - first_spot_grid_points (list of int): A list of grid points for the bright spots on the x-axis for each individual image.
    - spot_proportional_size (int): The proportional size of the bright spots (number of grid cells).
    - image_size (tuple): The size of each individual image (width, height).
    - grid_size (tuple): The size of the grid (columns, rows).
    - spot_color (tuple): The color of the bright spots (R, G, B).
    - padding (int): The padding (in pixels) to add between each image.
    - background_color (tuple): The background color for the padding (default: white).
    - text_color (tuple): The color of the text (default: black).
    - font_size (int): Font size for the text label on each rectangle.
    - show_text (bool): Whether or not to display the text above each spot.
    - num_columns (int): Number of images to display in each row.
    - txt_spacing (int): Additional space between the text and the image.
    - min_padding_above_text (int): Minimum space to maintain between the top of the text and the top of the image.
    - image_background_color (tuple): Background color for each individual image (default: white).
    - border (bool): Whether to add a border around each individual image.
    
    Returns:
    - PIL.Image: The combined image with all individual images arranged in multiple rows.
    """
    
    num_images = len(first_spot_grid_points)
    num_rows = (num_images + num_columns - 1) // num_columns  # Calculate how many rows are needed

    # Calculate the total height of the image, including padding and space for text
    text_height = font_size + txt_spacing + min_padding_above_text
    combined_image_height = (image_size[1] + (text_height if show_text else 0)) * num_rows + padding * (num_rows + 1)
    combined_image_width = image_size[0] * num_columns + padding * (num_columns + 1)

    # Create the base combined image
    combined_image = Image.new("RGB", (combined_image_width, combined_image_height), background_color)
    
    # Use a TrueType font if available
    try:
        font_path = "/Library/Fonts/Arial.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("TrueType font not found. Using default bitmap font.")
        font = ImageFont.load_default()

    for idx, spot_grid_point in enumerate(first_spot_grid_points):
        # Create each individual image with specified background color
        image = Image.new("RGB", image_size, image_background_color)
        draw = ImageDraw.Draw(image)

        # Calculate grid cell size
        grid_cell_width = image_size[0] // grid_size[0]
        grid_cell_height = image_size[1] // grid_size[1]

        # Calculate the size of the bright spot based on the proportional size
        spot_width = int(grid_cell_width * spot_proportional_size)
        spot_height = int(grid_cell_height * spot_proportional_size)

        # Ensure the starting grid point keeps the spot inside the image
        spot_grid_point_adjusted = min(spot_grid_point, grid_size[0] - 1)

        # Calculate the starting position for the bright spot on the x-axis
        first_spot_x = spot_grid_point_adjusted * grid_cell_width
        y = (image_size[1] - spot_height) // 2

        # Add a single bright spot at the defined grid point
        draw.rectangle([first_spot_x, y, first_spot_x + spot_width, y + spot_height], fill=spot_color)

        # Add border if specified
        if border:
            draw.rectangle([0, 0, image_size[0] - 1, image_size[1] - 1], outline="black")

        # Calculate the x and y position for this image in the combined grid
        col = idx % num_columns  # Current column
        row = idx // num_columns  # Current row
        
        x_position = col * image_size[0] + padding * (col + 1)
        y_position = row * (image_size[1] + text_height if show_text else 0) + padding * (row + 1)

        # Paste the individual image into the combined image
        combined_image.paste(image, (x_position, y_position))

        # Conditionally add text if show_text is True
        if show_text:
            # Use index + 1 as the text label
            text = f"{idx + 1}"

            # Use textbbox to get the bounding box of the text
            draw_combined = ImageDraw.Draw(combined_image)
            text_bbox = draw_combined.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Calculate text size and position it centered in the padding area above the rectangle
            text_x = x_position + (image_size[0] - text_width) // 2
            text_y = max(min_padding_above_text, y_position - (padding // 2 + txt_spacing))

            # Draw the text label
            draw_combined.text((text_x, text_y), text, fill=text_color, font=font)

    return combined_image

def add_subplot_label(ax, label_text, xpos=-0.1, ypos=1.1, fontsize=16, 
                     fontweight='bold', ha='center', va='center', 
                     transform_type='transAxes', force_show=False, **kwargs):
    """
    Universal subplot labeling function that provides consistent labeling
    across all figures while preserving exact positioning and formatting.
    
    This function replaces all current labeling methods:
    1. Direct text() calls
    2. label_axis() with Roman numerals  
    3. Array-based panel labels
    4. Sequential capital letters
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to add the label to
    label_text : str
        The label text (e.g., 'A', 'B', 'Ai', 'Bii', etc.)
    xpos : float, default=-0.1
        X position for the label
    ypos : float, default=1.1
        Y position for the label
    fontsize : int, default=16
        Font size for the label
    fontweight : str, default='bold'
        Font weight for the label
    ha : str, default='center'
        Horizontal alignment
    va : str, default='center'
        Vertical alignment
    transform_type : str, default='transAxes'
        Transform type ('transAxes' or 'transData')
    force_show : bool, default=False
        If True, show label regardless of global SUBPLOT_LABELS_ENABLED setting
    **kwargs : dict
        Additional matplotlib text parameters
    
    Returns:
    --------
    matplotlib.text.Text or None
        The created text object if labels are enabled, None otherwise
    
    Example Usage:
    --------------
    # Replace: bpf.add_subplot_label(ax, 'A', xpos=0.05, ypos=1.1, fontsize=16, fontweight='bold', ha='center', va='center')
    # With: add_subplot_label(ax, 'A', xpos=0.05, ypos=1.1)
    
    # Replace: label_axis([ax1, ax2], "H") 
    # With: add_subplot_label(ax1, 'Hi', xpos=-0.1, ypos=1.1)
    #       add_subplot_label(ax2, 'Hii', xpos=-0.1, ypos=1.1)
    
    # Control visibility globally:
    # bpf.set_subplot_labels_enabled(False)  # Turn off all labels
    # bpf.set_subplot_labels_enabled(True)   # Turn on all labels
    """
    # Check global flag unless force_show is True
    if not force_show and not SUBPLOT_LABELS_ENABLED:
        return None
    
    # Set transform based on transform_type
    if transform_type == 'transAxes':
        transform = ax.transAxes
    elif transform_type == 'transData':
        transform = ax.transData
    else:
        transform = ax.transAxes  # default fallback
    
    # Create the text label with all specified parameters
    text_obj = ax.text(xpos, ypos, label_text,
                      transform=transform,
                      fontsize=fontsize,
                      fontweight=fontweight,
                      ha=ha,
                      va=va,
                      **kwargs)
    
    return text_obj


def add_subplot_labels_from_list(axes_list, labels_list, base_params=None, 
                                individual_params=None, force_show=False):
    """
    Apply labels to a list of axes with consistent or individual parameters.
    
    This function is designed to replace label_axis() calls while maintaining
    exact positioning and formatting.
    
    Parameters:
    -----------
    axes_list : list
        List of matplotlib axes objects
    labels_list : list
        List of label strings (e.g., ['Hi', 'Hii', 'Hiii'])
    base_params : dict, optional
        Base parameters applied to all labels (xpos, ypos, fontsize, etc.)
    individual_params : list of dict, optional
        Individual parameters for each label (overrides base_params)
    force_show : bool, default=False
        If True, show labels regardless of global SUBPLOT_LABELS_ENABLED setting
    
    Returns:
    --------
    list
        List of created text objects (may contain None values if labels disabled)
    
    Example Usage:
    --------------
    # Replace: h_axes = axs_mini_list
    h_labels = bpf.generate_letter_roman_labels('H', len(h_axes))
    bpf.add_subplot_labels_from_list(h_axes, h_labels, 
                                base_params={'xpos': -0.1, 'ypos': 1.1, 'fontsize': 16, 'fontweight': 'bold'})
    # With: add_subplot_labels_from_list(axs_mini_list, ['Hi', 'Hii'], 
    #                                   base_params={'xpos': -0.1, 'ypos': 1.1})
    """
    if base_params is None:
        base_params = {}
    
    if individual_params is None:
        individual_params = [{}] * len(axes_list)
    
    text_objects = []
    
    for i, (ax, label) in enumerate(zip(axes_list, labels_list)):
        # Merge base params with individual params for this axis
        params = {**base_params}
        if i < len(individual_params):
            params.update(individual_params[i])
        
        # Add force_show parameter to the params
        params['force_show'] = force_show
        
        text_obj = add_subplot_label(ax, label, **params)
        text_objects.append(text_obj)
    
    return text_objects


def int_to_roman(num):
    """
    Convert integer to Roman numeral (lowercase).
    Kept for backward compatibility and Roman numeral generation.
    
    Parameters:
    -----------
    num : int
        Number to convert (1-100)
        
    Returns:
    --------
    str
        Roman numeral representation
    """
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
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


def generate_letter_roman_labels(letter, count):
    """
    Generate a list of letter + Roman numeral combinations.
    
    Parameters:
    -----------
    letter : str
        Base letter (e.g., 'A', 'B', 'H')
    count : int
        Number of labels to generate
        
    Returns:
    --------
    list
        List of labels (e.g., ['Ai', 'Aii', 'Aiii'])
    
    Example:
    --------
    >>> generate_letter_roman_labels('H', 3)
    ['Hi', 'Hii', 'Hiii']
    """
    return [f"{letter}{int_to_roman(i+1)}" for i in range(count)]


def set_subplot_labels_enabled(enabled):
    """
    Set global flag to enable or disable all subplot labels.
    
    Parameters:
    -----------
    enabled : bool
        True to enable labels, False to disable labels
    
    Example Usage:
    --------------
    # Turn off all subplot labels
    bpf.set_subplot_labels_enabled(False)
    
    # Turn on all subplot labels  
    bpf.set_subplot_labels_enabled(True)
    """
    global SUBPLOT_LABELS_ENABLED
    SUBPLOT_LABELS_ENABLED = enabled
    print(f"✓ Subplot labels {'ENABLED' if enabled else 'DISABLED'} globally")


def get_subplot_labels_enabled():
    """
    Get current state of global subplot labels flag.
    
    Returns:
    --------
    bool
        Current state of SUBPLOT_LABELS_ENABLED
    
    Example Usage:
    --------------
    if bpf.get_subplot_labels_enabled():
        print("Labels are currently enabled")
    else:
        print("Labels are currently disabled")
    """
    return SUBPLOT_LABELS_ENABLED


def toggle_subplot_labels():
    """
    Toggle the global subplot labels flag between enabled and disabled.
    
    Returns:
    --------
    bool
        New state after toggling
    
    Example Usage:
    --------------
    # Toggle labels on/off
    new_state = bpf.toggle_subplot_labels()
    print(f"Labels are now {'ON' if new_state else 'OFF'}")
    """
    global SUBPLOT_LABELS_ENABLED
    SUBPLOT_LABELS_ENABLED = not SUBPLOT_LABELS_ENABLED
    print(f"✓ Subplot labels toggled to {'ENABLED' if SUBPLOT_LABELS_ENABLED else 'DISABLED'}")
    return SUBPLOT_LABELS_ENABLED

# Global figure saving control
# Check environment variable first, then default to 'png'
_env_format = os.environ.get('FIGURE_FORMAT', 'png')
FIGURE_FORMAT = _env_format.lower()

# Check for multiple formats
_env_formats = os.environ.get('FIGURE_FORMATS', '')
FIGURE_FORMATS = [fmt.strip().lower() for fmt in _env_formats.split(',') if fmt.strip()] if _env_formats else []

# Check for DPI setting
_env_dpi = os.environ.get('FIGURE_DPI', '300')
FIGURE_DPI = int(_env_dpi) if _env_dpi.isdigit() else 300

# Check for transparency setting
_env_transparent = os.environ.get('FIGURE_TRANSPARENT', 'False')
FIGURE_TRANSPARENT = _env_transparent.lower() in ('true', '1', 'yes', 'on')

# Quality settings for different formats (dynamically updated from environment)
def _get_base_quality_settings():
    """Get base quality settings with current environment values."""
    return {
        'dpi': FIGURE_DPI,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white' if not FIGURE_TRANSPARENT else 'none',
        'edgecolor': 'none',
        'transparent': FIGURE_TRANSPARENT
    }

FIGURE_QUALITY_SETTINGS = {
    'png': _get_base_quality_settings(),
    'pdf': _get_base_quality_settings(),
    'svg': _get_base_quality_settings(),
    'eps': _get_base_quality_settings()
}

def set_figure_format(format_type):
    """
    Set the global figure format for all plots.
    
    Parameters:
    - format_type (str): The format to save figures in ('png', 'pdf', 'svg', 'eps')
    """
    global FIGURE_FORMAT
    if format_type.lower() in FIGURE_QUALITY_SETTINGS:
        FIGURE_FORMAT = format_type.lower()
        print(f"Figure format set to: {FIGURE_FORMAT.upper()}")
    else:
        print(f"Warning: Unsupported format '{format_type}'. Using default 'png'.")
        FIGURE_FORMAT = 'png'

def get_figure_format():
    """
    Get the current global figure format.
    
    Returns:
    - str: Current figure format
    """
    return FIGURE_FORMAT

def save_figure_smart(fig, output_dir, filename, create_dir=True, verbose=True):
    """
    Smart figure saving that automatically handles global format settings.
    Uses FIGURE_FORMATS if set, otherwise uses FIGURE_FORMAT.
    
    Parameters:
    - fig: matplotlib figure object
    - output_dir (str): Directory to save the figure
    - filename (str): Base filename (without extension)
    - create_dir (bool): Create output directory if it doesn't exist
    - verbose (bool): Print save confirmations
    
    Returns:
    - list: List of full paths of saved files
    """
    if FIGURE_FORMATS:
        # Multiple formats requested
        return save_figure_multiple_formats(fig, output_dir, filename, 
                                          formats=FIGURE_FORMATS,
                                          create_dir=create_dir, verbose=verbose)
    else:
        # Single format
        saved_path = save_figure(fig, output_dir, filename, 
                               format_override=FIGURE_FORMAT,
                               create_dir=create_dir, verbose=verbose)
        return [saved_path]

def save_figure(fig, output_dir, filename, format_override=None, quality_override=None, 
                create_dir=True, verbose=True):
    """
    Unified figure saving function with consistent quality settings.
    
    Parameters:
    - fig: matplotlib figure object
    - output_dir (str): Directory to save the figure
    - filename (str): Base filename (without extension)
    - format_override (str): Override global format for this save
    - quality_override (dict): Override quality settings for this save
    - create_dir (bool): Create output directory if it doesn't exist
    - verbose (bool): Print save confirmation
    
    Returns:
    - str: Full path of saved file
    """
    # Determine format to use
    format_to_use = format_override.lower() if format_override else FIGURE_FORMAT
    
    # Ensure format is supported
    if format_to_use not in FIGURE_QUALITY_SETTINGS:
        print(f"Warning: Unsupported format '{format_to_use}'. Using 'png'.")
        format_to_use = 'png'
    
    # Create output directory if needed
    if create_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created directory: {output_dir}")
    
    # Build full output path
    full_filename = f"{filename}.{format_to_use}"
    output_path = os.path.join(output_dir, full_filename)
    
    # Get quality settings (refresh from environment)
    quality_settings = _get_base_quality_settings().copy()
    if quality_override:
        quality_settings.update(quality_override)
    
    # Save the figure
    fig.savefig(output_path, format=format_to_use, **quality_settings)
    
    if verbose:
        print(f"Figure saved: {output_path}")
    
    return output_path

def save_figure_multiple_formats(fig, output_dir, filename, formats=['png'], 
                                quality_override=None, create_dir=True, verbose=True):
    """
    Save figure in multiple formats simultaneously.
    
    Parameters:
    - fig: matplotlib figure object
    - output_dir (str): Directory to save the figure
    - filename (str): Base filename (without extension)
    - formats (list): List of formats to save ('png', 'pdf', 'svg', 'eps')
    - quality_override (dict): Override quality settings for all saves
    - create_dir (bool): Create output directory if it doesn't exist
    - verbose (bool): Print save confirmations
    
    Returns:
    - list: List of full paths of saved files
    """
    saved_files = []
    
    for fmt in formats:
        try:
            saved_path = save_figure(fig, output_dir, filename, 
                                   format_override=fmt, 
                                   quality_override=quality_override,
                                   create_dir=create_dir, 
                                   verbose=verbose)
            saved_files.append(saved_path)
        except Exception as e:
            print(f"Error saving {fmt} format: {e}")
    
    return saved_files

def get_figure_quality_settings(format_type=None):
    """
    Get quality settings for a specific format.
    
    Parameters:
    - format_type (str): Format to get settings for (defaults to current global format)
    
    Returns:
    - dict: Quality settings for the format
    """
    format_to_use = format_type.lower() if format_type else FIGURE_FORMAT
    return FIGURE_QUALITY_SETTINGS.get(format_to_use, FIGURE_QUALITY_SETTINGS['png']).copy()

def update_figure_quality_settings(format_type, **kwargs):
    """
    Update quality settings for a specific format.
    
    Parameters:
    - format_type (str): Format to update settings for
    - **kwargs: Quality settings to update
    """
    if format_type.lower() in FIGURE_QUALITY_SETTINGS:
        FIGURE_QUALITY_SETTINGS[format_type.lower()].update(kwargs)
        print(f"Updated {format_type.upper()} quality settings: {kwargs}")
    else:
        print(f"Warning: Unknown format '{format_type}'")

# Convenience function for backward compatibility
def save_plot(fig, outpath, **kwargs):
    """
    Backward compatibility function for existing scripts.
    Automatically detects format from file extension.
    
    Parameters:
    - fig: matplotlib figure object
    - outpath (str): Full output path including extension
    - **kwargs: Additional arguments passed to savefig
    """
    # Extract directory, filename, and extension
    output_dir = os.path.dirname(outpath)
    full_filename = os.path.basename(outpath)
    filename, ext = os.path.splitext(full_filename)
    
    # Remove leading dot from extension
    format_detected = ext.lstrip('.').lower() if ext else 'png'
    
    # Use the unified save function
    return save_figure(fig, output_dir, filename, 
                      format_override=format_detected, 
                      create_dir=True, verbose=True)

