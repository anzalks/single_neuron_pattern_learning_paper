#!/usr/bin/env python3
"""
Shared Utilities Module for Pattern Learning Paper
Consolidates common functions from all plotting scripts to eliminate redundancy

Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/
Copyright 2024-, Anzal KS
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image, ImageDraw, ImageFont
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import h5py
from datetime import datetime

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get_data_path(self, key: str) -> str:
        """Get absolute data path from config"""
        base_dir = self.config['data_paths']['base_data_dir']
        
        # Check in analysis_pickles first
        if key in self.config['data_paths']['analysis_pickles']:
            rel_path = self.config['data_paths']['analysis_pickles'][key]
        elif key in self.config['data_paths']['illustrations']:
            rel_path = self.config['data_paths']['illustrations'][key]
        elif key == 'cell_stats':
            rel_path = self.config['data_paths']['cell_stats']
        else:
            raise KeyError(f"Data path key '{key}' not found in configuration")
        
        # Handle absolute paths
        if os.path.isabs(rel_path):
            return rel_path
        
        return os.path.join(base_dir, rel_path)
    
    def get_output_dir(self, figure_type: str = "main_figures") -> str:
        """Get output directory path"""
        base_output = self.config['output']['base_output_dir']
        figure_dir = self.config['output']['directories'][figure_type]
        return os.path.join(base_output, figure_dir)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup centralized logging"""
    logger = logging.getLogger('pattern_learning')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class DataManager:
    """Centralized data loading and validation"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger('pattern_learning.data')
        self._data_cache = {}
    
    def load_pickle(self, data_key: str, use_cache: bool = True) -> Any:
        """Load pickle file with compatibility handling"""
        file_path = self.config.get_data_path(data_key)
        
        if use_cache and data_key in self._data_cache:
            self.logger.debug(f"Loading from cache: {data_key}")
            return self._data_cache[data_key]
        
        try:
            self.logger.info(f"Loading pickle: {file_path}")
            
            # Try standard pickle loading first
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            except (ModuleNotFoundError, AttributeError) as e:
                if 'pandas' in str(e):
                    # Handle pandas compatibility issues
                    self.logger.warning(f"Pandas compatibility issue, trying alternative loading: {e}")
                    try:
                        # Try loading with ignore_unknown_types
                        import pandas as pd
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with open(file_path, 'rb') as f:
                                # Try to load with protocol compatibility
                                data = pickle.load(f, encoding='latin1')
                    except Exception:
                        # Last resort: try using joblib if available
                        try:
                            import joblib
                            self.logger.warning("Trying joblib.load as fallback")
                            data = joblib.load(file_path)
                        except ImportError:
                            # If joblib not available, raise original error
                            raise e
                else:
                    raise e
            
            if use_cache:
                self._data_cache[data_key] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading pickle file {file_path}: {e}")
            raise ValueError(f"Error loading pickle file {file_path}: {e}")
    
    def load_hdf5(self, data_key: str) -> Any:
        """Load HDF5 file"""
        file_path = self.config.get_data_path(data_key)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        self.logger.info(f"Loading HDF5: {file_path}")
        try:
            return pd.read_hdf(file_path)
        except Exception as e:
            raise ValueError(f"Error loading HDF5 file {file_path}: {e}")
    
    def load_image(self, data_key: str) -> Image.Image:
        """Load image file"""
        file_path = self.config.get_data_path(data_key)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        self.logger.info(f"Loading image: {file_path}")
        try:
            return Image.open(file_path)
        except Exception as e:
            raise ValueError(f"Error loading image file {file_path}: {e}")
    
    def validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate that all required data files exist"""
        missing_files = []
        for dep in dependencies:
            try:
                path = self.config.get_data_path(dep)
                if not os.path.exists(path):
                    missing_files.append(f"{dep}: {path}")
            except KeyError:
                missing_files.append(f"{dep}: key not found in config")
        
        if missing_files:
            self.logger.error(f"Missing dependencies: {missing_files}")
            return False
        
        self.logger.info("All dependencies validated successfully")
        return True

# =============================================================================
# PLOTTING UTILITIES (Consolidated from basic_plot_functions_and_features.py)
# =============================================================================

# Color definitions from the original baisic_plot_fuctnions_and_features
pre_color = "#000000"  # Black (original)
post_color = "#377eb8"  # Blue (original)
post_late = "#de6f00"  # Orange (original)

# Colorblind-friendly color cycle (original order)
CB_color_cycle = [
    '#377eb8',  # Blue
    '#ff7f00',  # Orange  
    '#4daf4a',  # Green
    '#f781bf',  # Pink
    '#a65628',  # Brown
    '#984ea3',  # Purple
    '#999999',  # Gray
    '#e41a1c',  # Red
    '#dede00'   # Yellow
]

def color_fader(c1: str, c2: str, mix: float = 0) -> str:
    """Fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)"""
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def convert_pvalue_to_asterisks(pvalue: float) -> str:
    """Convert p-value to significance asterisks"""
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    else:
        return "ns"

def set_plot_properties(config: Optional[Dict] = None):
    """Set standardized plot properties"""
    if config is None:
        # Default settings
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['legend.title_fontsize'] = 11
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['text.usetex'] = False
    else:
        # Use config settings
        plot_config = config.get('analysis', {}).get('plotting', {})
        plt.rcParams['figure.figsize'] = plot_config.get('figure_size', [12, 8])
        plt.rcParams['figure.dpi'] = plot_config.get('dpi', 300)
        plt.rcParams['font.size'] = plot_config.get('font_size', 12)

def subtract_baseline(trace: np.ndarray, sampling_rate: int = 20000, 
                     bl_period_in_ms: int = 5) -> np.ndarray:
    """Subtract baseline from trace"""
    bl_period = bl_period_in_ms / 1000
    bl_duration = int(sampling_rate * bl_period)
    bl = np.mean(trace[:bl_duration])
    return trace - bl

def map_points_to_patterns(pattern: str) -> Optional[List[str]]:
    """Map pattern names to point lists"""
    pattern_map = {
        'pattern_0': ['point_0', 'point_1', 'point_2', 'point_3', 'point_4'],
        'pattern_1': ['point_2', 'point_3', 'point_4', 'point_5', 'point_6'],
        'pattern_2': ['point_7', 'point_8', 'point_9', 'point_10', 'point_11']
    }
    
    if pattern not in pattern_map:
        logging.warning(f"Pattern {pattern} not recognized")
        return None
    
    return pattern_map[pattern]

def create_grid_image(first_spot_grid_point: int, spot_proportional_size: float = 1.5,
                     image_size: Tuple[int, int] = (1024, 480), 
                     grid_size: Tuple[int, int] = (24, 24), 
                     num_spots: int = 5, spot_color: Tuple[int, int, int] = (0, 0, 0),
                     background_color: Tuple[int, int, int] = (255, 255, 255),
                     border: bool = True) -> Image.Image:
    """Create a custom grid image with spots"""
    border_thickness = 5 if border else 0
    bordered_image_size = (image_size[0] + 2 * border_thickness, 
                          image_size[1] + 2 * border_thickness)
    
    # Create image with border
    image = Image.new("RGB", bordered_image_size, 
                     (0, 0, 0) if border else background_color)
    
    # Inner area for grid
    inner_image = Image.new("RGB", image_size, background_color)
    inner_draw = ImageDraw.Draw(inner_image)
    
    # Calculate grid dimensions
    grid_cell_width = image_size[0] // grid_size[0]
    grid_cell_height = image_size[1] // grid_size[1]
    
    spot_width = grid_cell_width * spot_proportional_size
    spot_height = grid_cell_height * spot_proportional_size
    
    # Position spots
    first_spot_x = first_spot_grid_point * grid_cell_width
    y = (image_size[1] - spot_height) // 2
    gap_size = spot_width // 2
    
    for i in range(num_spots):
        x = first_spot_x + i * (spot_width + gap_size)
        inner_draw.rectangle([x, y, x + spot_width, y + spot_height], 
                           fill=spot_color)
    
    # Paste inner image into bordered image
    if border:
        image.paste(inner_image, (border_thickness, border_thickness))
    else:
        image = inner_image
    
    return image

# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

class OutputManager:
    """Manage figure outputs and file organization"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger('pattern_learning.output')
    
    def create_output_dirs(self):
        """Create necessary output directories"""
        base_dir = self.config.config['output']['base_output_dir']
        dirs = self.config.config['output']['directories']
        
        for dir_name in dirs.values():
            full_path = os.path.join(base_dir, dir_name)
            os.makedirs(full_path, exist_ok=True)
            self.logger.debug(f"Created directory: {full_path}")
    
    def get_output_path(self, output_type: str, filename: str = None) -> str:
        """Get output path for different types of outputs"""
        base_dir = self.config.config['output']['base_output_dir']
        dirs = self.config.config['output']['directories']
        
        if output_type not in dirs:
            raise KeyError(f"Output type '{output_type}' not found in configuration")
        
        output_dir = os.path.join(base_dir, dirs[output_type])
        
        if filename:
            return os.path.join(output_dir, filename)
        return output_dir
    
    def save_figure(self, fig: plt.Figure, figure_name: str, 
                   figure_type: str = "main_figures", 
                   analysis_type: str = "standard") -> List[str]:
        """Save figure in proper directory structure"""
        
        # Determine the correct base directory
        if figure_name.startswith('supp_') or figure_type == "supplementary_figures":
            base_dir = self.get_output_path("supplementary_figures")
            # Extract supplementary figure number
            if figure_name.startswith('supp_figure_'):
                figure_num = figure_name.replace('supp_figure_', 'S')
            else:
                figure_num = figure_name.replace('figure_', 'S')
        else:
            base_dir = self.get_output_path("main_figures") 
            # Extract main figure number
            if figure_name.startswith('figure_'):
                figure_num = figure_name.replace('figure_', '')
            else:
                figure_num = figure_name
        
        # Create specific output directory: Figure_N or Figure_N_fnorm
        if analysis_type == "field_normalized":
            output_dir = os.path.join(base_dir, f"Figure_{figure_num}_fnorm")
        else:
            output_dir = os.path.join(base_dir, f"Figure_{figure_num}")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        formats = self.config.config['output']['figure_formats']
        dpi = self.config.config['output']['figure_dpi']
        
        saved_files = []
        
        for fmt in formats:
            # Simple filename: figure_N.ext
            filename = f"figure_{figure_num}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            
            try:
                fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                saved_files.append(filepath)
                self.logger.info(f"Saved figure: {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving figure {filepath}: {e}")
        
        return saved_files
    
    def save_health_stats(self, stats_data: Any, cell_id: str, 
                         analysis_type: str = "standard", 
                         file_format: str = "pickle") -> str:
        """Save health statistics with proper naming"""
        output_dir = self.get_output_path("health_stats")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use the template from config
        filename_template = self.config.config['output']['health_stats_template']
        filename = filename_template.format(
            cell_id=cell_id,
            analysis_type=analysis_type,
            timestamp=timestamp
        ) + f".{file_format}"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            if file_format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(stats_data, f)
            elif file_format == "h5":
                if isinstance(stats_data, pd.DataFrame):
                    stats_data.to_hdf(filepath, key='health_stats', mode='w')
                else:
                    raise ValueError("HDF5 format requires DataFrame input")
            elif file_format == "csv":
                if isinstance(stats_data, pd.DataFrame):
                    stats_data.to_csv(filepath, index=False)
                else:
                    raise ValueError("CSV format requires DataFrame input")
            
            self.logger.info(f"Saved health stats: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving health stats {filepath}: {e}")
            raise
    
    def save_cell_data(self, cell_data: Any, cell_id: str, 
                      analysis_type: str = "standard", 
                      file_format: str = "h5") -> str:
        """Save cell data with proper naming"""
        output_dir = self.get_output_path("cell_data_hdf")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use the template from config
        filename_template = self.config.config['output']['cell_data_template']
        filename = filename_template.format(
            cell_id=cell_id,
            analysis_type=analysis_type,
            timestamp=timestamp
        ) + f".{file_format}"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            if file_format == "h5":
                if isinstance(cell_data, pd.DataFrame):
                    cell_data.to_hdf(filepath, key='cell_data', mode='w')
                elif isinstance(cell_data, dict):
                    # Save dictionary as multiple keys in HDF5
                    with pd.HDFStore(filepath, mode='w') as store:
                        for key, value in cell_data.items():
                            if isinstance(value, pd.DataFrame):
                                store[key] = value
                            elif isinstance(value, np.ndarray):
                                # Convert numpy arrays to DataFrame
                                df = pd.DataFrame(value)
                                store[key] = df
                else:
                    raise ValueError("HDF5 format requires DataFrame or dict input")
            elif file_format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(cell_data, f)
            
            self.logger.info(f"Saved cell data: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving cell data {filepath}: {e}")
            raise
    
    def save_analysis_results(self, results: Any, analysis_name: str,
                            analysis_type: str = "standard",
                            file_format: str = "pickle") -> str:
        """Save analysis results with proper naming"""
        output_dir = self.get_output_path("analysis_results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{analysis_name}_{analysis_type}_{timestamp}.{file_format}"
        filepath = os.path.join(output_dir, filename)
        
        try:
            if file_format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
            elif file_format == "json":
                import json
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved analysis results: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results {filepath}: {e}")
            raise

# =============================================================================
# ANALYSIS TYPE DETECTION
# =============================================================================

def detect_analysis_type(script_path: str) -> str:
    """Detect analysis type from script filename"""
    if 'fnorm' in script_path.lower():
        return 'field_normalized'
    return 'standard'

def get_classification_key(analysis_type: str) -> str:
    """Get appropriate classification pickle key based on analysis type"""
    if analysis_type == 'field_normalized':
        return 'all_cells_fnorm_classified_dict'
    return 'all_cells_classified_dict'

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_data_integrity(data: Any, data_type: str) -> bool:
    """Validate data integrity based on type"""
    if data is None:
        return False
    
    if data_type == 'dataframe' and isinstance(data, pd.DataFrame):
        return not data.empty
    elif data_type == 'dict' and isinstance(data, dict):
        return len(data) > 0
    elif data_type == 'array' and isinstance(data, np.ndarray):
        return data.size > 0
    
    return True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir_exists(directory: str):
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)

def cleanup_legacy_files(directory: str, patterns: List[str]):
    """Clean up legacy files matching patterns"""
    logger = logging.getLogger('pattern_learning.cleanup')
    
    for pattern in patterns:
        # This would need glob for pattern matching
        # For now, just log what would be cleaned
        logger.info(f"Would clean files matching {pattern} in {directory}")

# =============================================================================
# MAIN UTILITIES CLASS
# =============================================================================

class PatternLearningUtils:
    """Main utilities class that combines all managers"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.data_manager = DataManager(self.config_manager)
        self.output_manager = OutputManager(self.config_manager)
        self.logger = setup_logging(
            self.config_manager.config['pipeline']['log_level']
        )
        
        # Set plot properties
        set_plot_properties(self.config_manager.config)
        
        # Create output directories
        self.output_manager.create_output_dirs()
    
    def get_figure_config(self, figure_name: str) -> Dict[str, Any]:
        """Get configuration for specific figure"""
        figures_config = self.config_manager.config['figures']
        
        # Check main figures
        if figure_name in figures_config['main_figures']:
            return figures_config['main_figures'][figure_name]
        
        # Check supplementary figures
        if figure_name in figures_config['supplementary_figures']:
            return figures_config['supplementary_figures'][figure_name]
        
        raise KeyError(f"Figure {figure_name} not found in configuration")
    
    def load_figure_data(self, figure_name: str, analysis_type: str = "standard") -> Dict[str, Any]:
        """Load all required data for a figure"""
        figure_config = self.get_figure_config(figure_name)
        dependencies = figure_config['data_dependencies']
        
        # Validate dependencies
        if not self.data_manager.validate_dependencies(dependencies):
            raise ValueError(f"Missing dependencies for {figure_name}")
        
        # Load data with improved type detection
        data = {}
        for dep in dependencies:
            try:
                # Get the actual file path to check extension
                file_path = self.config_manager.get_data_path(dep)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if dep == 'cell_stats' or file_ext in ['.h5', '.hdf5']:
                    # HDF5 files
                    data[dep] = self.data_manager.load_hdf5(dep)
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    # Image files
                    data[dep] = self.data_manager.load_image(dep)
                elif file_ext in ['.pickle', '.pkl'] or dep.endswith('_dict') or dep.startswith('pd_') or dep.startswith('all_') or 'traces' in dep:
                    # Pickle files - handle field normalized analysis
                    if dep == 'all_cells_classified_dict' and analysis_type == 'field_normalized':
                        data[dep] = self.data_manager.load_pickle('all_cells_fnorm_classified_dict')
                    else:
                        data[dep] = self.data_manager.load_pickle(dep)
                else:
                    # Default to pickle for unknown extensions or legacy data
                    self.logger.warning(f"Unknown file type for {dep} ({file_path}), attempting pickle loader")
                    data[dep] = self.data_manager.load_pickle(dep)
                    
            except Exception as e:
                self.logger.error(f"Error loading {dep}: {e}")
                raise
        
        return data 

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
    image_background_color=(255, 255, 255),
    border=True
):
    """
    Creates a single image composed of multiple individual images arranged in multiple rows. 
    Optionally, text is shown above each spot.
    
    Parameters:
    - first_spot_grid_points (list of int): A list of grid points for the bright spots
    - spot_proportional_size (int): The proportional size of the bright spots
    - image_size (tuple): The size of each individual image (width, height)
    - grid_size (tuple): The size of the grid (columns, rows)
    - spot_color (tuple): The color of the bright spots (R, G, B)
    - padding (int): The padding (in pixels) to add between each image
    - background_color (tuple): The background color for the padding
    - text_color (tuple): The color of the text
    - font_size (int): Font size for the text label
    - show_text (bool): Whether to display text above each spot
    - num_columns (int): Number of images to display in each row
    - txt_spacing (int): Additional space between text and image
    - min_padding_above_text (int): Minimum space above text
    - image_background_color (tuple): Background color for each individual image
    - border (bool): Whether to add a border around each individual image
    
    Returns:
    - PIL.Image: The combined image with all individual images arranged in multiple rows
    """
    from PIL import Image, ImageDraw, ImageFont
    
    num_images = len(first_spot_grid_points)
    num_rows = (num_images + num_columns - 1) // num_columns

    # Calculate the total height of the image
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

        # Calculate the size of the bright spot
        spot_width = int(grid_cell_width * spot_proportional_size)
        spot_height = int(grid_cell_height * spot_proportional_size)

        # Ensure the starting grid point keeps the spot inside the image
        spot_grid_point_adjusted = min(spot_grid_point, grid_size[0] - 1)

        # Calculate the starting position for the bright spot
        first_spot_x = spot_grid_point_adjusted * grid_cell_width
        y = (image_size[1] - spot_height) // 2

        # Add a single bright spot at the defined grid point
        draw.rectangle([first_spot_x, y, first_spot_x + spot_width, y + spot_height], fill=spot_color)

        # Add border if specified
        if border:
            draw.rectangle([0, 0, image_size[0] - 1, image_size[1] - 1], outline="black")

        # Calculate position for this image in the combined grid
        col = idx % num_columns
        row = idx // num_columns
        
        x_position = col * image_size[0] + padding * (col + 1)
        y_position = row * (image_size[1] + text_height if show_text else 0) + padding * (row + 1)

        # Paste the individual image into the combined image
        combined_image.paste(image, (x_position, y_position))

        # Add text if show_text is True
        if show_text:
            text = f"{idx + 1}"

            # Get text dimensions
            draw_combined = ImageDraw.Draw(combined_image)
            text_bbox = draw_combined.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text centered above the image
            text_x = x_position + (image_size[0] - text_width) // 2
            text_y = max(min_padding_above_text, y_position - (padding // 2 + txt_spacing))

            # Draw the text label
            draw_combined.text((text_x, text_y), text, fill=text_color, font=font)

    return combined_image 