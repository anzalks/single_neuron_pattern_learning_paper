#!/usr/bin/env python3
"""
Unit Tests for Shared Utilities Module
Tests all core functionality of the pattern learning pipeline

Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/
"""

import pytest
import os
import sys
import tempfile
import yaml
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared_utilities import (
    ConfigManager, DataManager, OutputManager, PatternLearningUtils,
    color_fader, convert_pvalue_to_asterisks, subtract_baseline,
    map_points_to_patterns, create_grid_image, detect_analysis_type,
    get_classification_key, validate_data_integrity
)

class TestConfigManager:
    """Test ConfigManager functionality"""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing"""
        return {
            'data_paths': {
                'base_data_dir': '/test/data',
                'analysis_pickles': {
                    'test_pickle': 'test.pickle'
                },
                'illustrations': {
                    'test_image': 'test.jpg'
                },
                'cell_stats': 'stats.h5'
            },
            'output': {
                'base_output_dir': './outputs',
                'directories': {
                    'main_figures': 'main',
                    'supplementary_figures': 'supp'
                }
            }
        }
    
    @pytest.fixture
    def config_file(self, sample_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name
    
    def test_load_config_success(self, config_file):
        """Test successful config loading"""
        config_manager = ConfigManager(config_file)
        assert config_manager.config is not None
        assert 'data_paths' in config_manager.config
        
        # Cleanup
        os.unlink(config_file)
    
    def test_load_config_file_not_found(self):
        """Test config loading with missing file"""
        with pytest.raises(FileNotFoundError):
            ConfigManager('nonexistent.yaml')
    
    def test_get_data_path_pickle(self, config_file):
        """Test getting data path for pickle files"""
        config_manager = ConfigManager(config_file)
        path = config_manager.get_data_path('test_pickle')
        assert path == '/test/data/test.pickle'
        
        # Cleanup
        os.unlink(config_file)
    
    def test_get_data_path_illustration(self, config_file):
        """Test getting data path for illustrations"""
        config_manager = ConfigManager(config_file)
        path = config_manager.get_data_path('test_image')
        assert path == '/test/data/test.jpg'
        
        # Cleanup
        os.unlink(config_file)
    
    def test_get_data_path_not_found(self, config_file):
        """Test getting data path for non-existent key"""
        config_manager = ConfigManager(config_file)
        with pytest.raises(KeyError):
            config_manager.get_data_path('nonexistent_key')
        
        # Cleanup
        os.unlink(config_file)
    
    def test_get_output_dir(self, config_file):
        """Test getting output directory"""
        config_manager = ConfigManager(config_file)
        output_dir = config_manager.get_output_dir('main_figures')
        assert output_dir == './outputs/main'
        
        # Cleanup
        os.unlink(config_file)

class TestDataManager:
    """Test DataManager functionality"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager"""
        mock_config = Mock()
        mock_config.get_data_path.return_value = '/test/path/data.pickle'
        return mock_config
    
    def test_data_manager_init(self, mock_config_manager):
        """Test DataManager initialization"""
        data_manager = DataManager(mock_config_manager)
        assert data_manager.config == mock_config_manager
        assert data_manager._data_cache == {}
    
    @patch('builtins.open')
    @patch('pickle.load')
    @patch('os.path.exists')
    def test_load_pickle_success(self, mock_exists, mock_pickle_load, mock_open, mock_config_manager):
        """Test successful pickle loading"""
        mock_exists.return_value = True
        mock_pickle_load.return_value = {'test': 'data'}
        
        data_manager = DataManager(mock_config_manager)
        result = data_manager.load_pickle('test_key')
        
        assert result == {'test': 'data'}
        mock_config_manager.get_data_path.assert_called_with('test_key')
    
    @patch('os.path.exists')
    def test_load_pickle_file_not_found(self, mock_exists, mock_config_manager):
        """Test pickle loading with missing file"""
        mock_exists.return_value = False
        
        data_manager = DataManager(mock_config_manager)
        with pytest.raises(FileNotFoundError):
            data_manager.load_pickle('test_key')
    
    @patch('pandas.read_hdf')
    @patch('os.path.exists')
    def test_load_hdf5_success(self, mock_exists, mock_read_hdf, mock_config_manager):
        """Test successful HDF5 loading"""
        mock_exists.return_value = True
        mock_read_hdf.return_value = pd.DataFrame({'test': [1, 2, 3]})
        
        data_manager = DataManager(mock_config_manager)
        result = data_manager.load_hdf5('test_key')
        
        assert isinstance(result, pd.DataFrame)
        mock_config_manager.get_data_path.assert_called_with('test_key')

class TestPlottingUtilities:
    """Test plotting utility functions"""
    
    def test_color_fader(self):
        """Test color fading function"""
        result = color_fader('#000000', '#ffffff', 0.5)
        assert isinstance(result, str)
        assert result.startswith('#')
    
    def test_convert_pvalue_to_asterisks(self):
        """Test p-value to asterisks conversion"""
        assert convert_pvalue_to_asterisks(0.00001) == "****"
        assert convert_pvalue_to_asterisks(0.0001) == "****"
        assert convert_pvalue_to_asterisks(0.001) == "***"
        assert convert_pvalue_to_asterisks(0.01) == "**"
        assert convert_pvalue_to_asterisks(0.05) == "*"
        assert convert_pvalue_to_asterisks(0.1) == "ns"
    
    def test_subtract_baseline(self):
        """Test baseline subtraction"""
        # Create test trace
        trace = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5])
        result = subtract_baseline(trace, sampling_rate=10, bl_period_in_ms=500)
        
        # First 5 samples should be baseline (mean = 1)
        # Result should be trace - 1
        expected = trace - 1
        np.testing.assert_array_equal(result, expected)
    
    def test_map_points_to_patterns(self):
        """Test pattern to points mapping"""
        result = map_points_to_patterns('pattern_0')
        expected = ['point_0', 'point_1', 'point_2', 'point_3', 'point_4']
        assert result == expected
        
        result = map_points_to_patterns('pattern_1')
        expected = ['point_2', 'point_3', 'point_4', 'point_5', 'point_6']
        assert result == expected
        
        result = map_points_to_patterns('pattern_2')
        expected = ['point_7', 'point_8', 'point_9', 'point_10', 'point_11']
        assert result == expected
        
        result = map_points_to_patterns('invalid_pattern')
        assert result is None
    
    def test_create_grid_image(self):
        """Test grid image creation"""
        image = create_grid_image(
            first_spot_grid_point=5,
            spot_proportional_size=1.0,
            image_size=(100, 50),
            grid_size=(10, 5),
            num_spots=3
        )
        
        assert isinstance(image, Image.Image)
        assert image.size == (110, 60)  # With border

class TestAnalysisUtilities:
    """Test analysis utility functions"""
    
    def test_detect_analysis_type(self):
        """Test analysis type detection"""
        assert detect_analysis_type('script_fnorm.py') == 'field_normalized'
        assert detect_analysis_type('script_FNORM.py') == 'field_normalized'
        assert detect_analysis_type('script.py') == 'standard'
        assert detect_analysis_type('script_v2.py') == 'standard'
    
    def test_get_classification_key(self):
        """Test classification key selection"""
        assert get_classification_key('field_normalized') == 'all_cells_fnorm_classified_dict'
        assert get_classification_key('standard') == 'all_cells_classified_dict'
        assert get_classification_key('other') == 'all_cells_classified_dict'
    
    def test_validate_data_integrity(self):
        """Test data integrity validation"""
        # Test DataFrame
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert validate_data_integrity(df, 'dataframe') == True
        
        empty_df = pd.DataFrame()
        assert validate_data_integrity(empty_df, 'dataframe') == False
        
        # Test dict
        test_dict = {'a': 1, 'b': 2}
        assert validate_data_integrity(test_dict, 'dict') == True
        
        empty_dict = {}
        assert validate_data_integrity(empty_dict, 'dict') == False
        
        # Test array
        arr = np.array([1, 2, 3])
        assert validate_data_integrity(arr, 'array') == True
        
        empty_arr = np.array([])
        assert validate_data_integrity(empty_arr, 'array') == False
        
        # Test None
        assert validate_data_integrity(None, 'dataframe') == False

class TestOutputManager:
    """Test OutputManager functionality"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager for OutputManager"""
        mock_config = Mock()
        mock_config.config = {
            'output': {
                'base_output_dir': './test_outputs',
                'directories': {
                    'main_figures': 'main',
                    'logs': 'logs'
                },
                'figure_formats': ['png', 'pdf'],
                'figure_dpi': 300
            }
        }
        mock_config.get_output_dir.return_value = './test_outputs/main'
        return mock_config
    
    def test_output_manager_init(self, mock_config_manager):
        """Test OutputManager initialization"""
        output_manager = OutputManager(mock_config_manager)
        assert output_manager.config == mock_config_manager
    
    @patch('os.makedirs')
    def test_create_output_dirs(self, mock_makedirs, mock_config_manager):
        """Test output directory creation"""
        output_manager = OutputManager(mock_config_manager)
        output_manager.create_output_dirs()
        
        # Should call makedirs for each directory
        assert mock_makedirs.call_count >= 1

class TestPatternLearningUtils:
    """Test main PatternLearningUtils class"""
    
    @pytest.fixture
    def sample_config_file(self):
        """Create a minimal config file for testing"""
        config = {
            'data_paths': {
                'base_data_dir': '/test',
                'analysis_pickles': {},
                'illustrations': {},
                'cell_stats': 'stats.h5'
            },
            'output': {
                'base_output_dir': './outputs',
                'directories': {'main_figures': 'main', 'logs': 'logs'},
                'figure_formats': ['png'],
                'figure_dpi': 300
            },
            'pipeline': {
                'log_level': 'INFO',
                'max_workers': 2
            },
            'figures': {
                'main_figures': {
                    'figure_1': {
                        'script': 'test.py',
                        'data_dependencies': ['test_data'],
                        'analysis_types': ['standard']
                    }
                },
                'supplementary_figures': {}
            },
            'analysis': {
                'plotting': {
                    'figure_size': [10, 6],
                    'dpi': 300,
                    'font_size': 12
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    @patch('src.shared_utilities.setup_logging')
    @patch('src.shared_utilities.set_plot_properties')
    def test_pattern_learning_utils_init(self, mock_set_plot, mock_setup_log, sample_config_file):
        """Test PatternLearningUtils initialization"""
        mock_logger = Mock()
        mock_setup_log.return_value = mock_logger
        
        utils = PatternLearningUtils(sample_config_file)
        
        assert utils.config_manager is not None
        assert utils.data_manager is not None
        assert utils.output_manager is not None
        
        # Cleanup
        os.unlink(sample_config_file)
    
    def test_get_figure_config_main(self, sample_config_file):
        """Test getting figure configuration for main figures"""
        with patch('src.shared_utilities.setup_logging'), \
             patch('src.shared_utilities.set_plot_properties'):
            
            utils = PatternLearningUtils(sample_config_file)
            config = utils.get_figure_config('figure_1')
            
            assert config['script'] == 'test.py'
            assert 'standard' in config['analysis_types']
        
        # Cleanup
        os.unlink(sample_config_file)
    
    def test_get_figure_config_not_found(self, sample_config_file):
        """Test getting configuration for non-existent figure"""
        with patch('src.shared_utilities.setup_logging'), \
             patch('src.shared_utilities.set_plot_properties'):
            
            utils = PatternLearningUtils(sample_config_file)
            
            with pytest.raises(KeyError):
                utils.get_figure_config('nonexistent_figure')
        
        # Cleanup
        os.unlink(sample_config_file)

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.fixture
    def complete_config(self):
        """Create a complete configuration for integration testing"""
        return {
            'data_paths': {
                'base_data_dir': '/test/data',
                'analysis_pickles': {
                    'pd_all_cells_mean': 'mean.pickle',
                    'all_cells_classified_dict': 'classified.pickle'
                },
                'illustrations': {
                    'figure_2_1': 'fig2_1.jpg'
                },
                'cell_stats': 'stats.h5'
            },
            'output': {
                'base_output_dir': './outputs',
                'directories': {
                    'main_figures': 'main',
                    'supplementary_figures': 'supp',
                    'logs': 'logs'
                },
                'figure_formats': ['png', 'pdf'],
                'figure_dpi': 300
            },
            'pipeline': {
                'log_level': 'INFO',
                'max_workers': 2,
                'timeout_per_figure': 300,
                'continue_on_error': False
            },
            'figures': {
                'main_figures': {
                    'figure_2': {
                        'script': 'plotting_scripts/main_figures/figure_generation_script_2.py',
                        'name': 'Test Figure 2',
                        'data_dependencies': ['pd_all_cells_mean', 'all_cells_classified_dict', 'figure_2_1'],
                        'analysis_types': ['standard', 'field_normalized']
                    }
                },
                'supplementary_figures': {}
            },
            'analysis': {
                'plotting': {
                    'figure_size': [12, 8],
                    'dpi': 300,
                    'font_size': 12
                }
            },
            'environment': {
                'python_executable': 'python',
                'required_packages': ['numpy', 'pandas', 'matplotlib']
            }
        }
    
    def test_end_to_end_config_loading(self, complete_config):
        """Test end-to-end configuration loading and validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(complete_config, f)
            config_file = f.name
        
        try:
            # Test config loading
            config_manager = ConfigManager(config_file)
            assert config_manager.config is not None
            
            # Test data path resolution
            path = config_manager.get_data_path('pd_all_cells_mean')
            assert path == '/test/data/mean.pickle'
            
            # Test output directory
            output_dir = config_manager.get_output_dir('main_figures')
            assert output_dir == './outputs/main'
            
        finally:
            os.unlink(config_file)

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 