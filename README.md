# single_neuron_pattern_learning_paper
All scripts that's associated with the pattern learning paper

## Recent Updates

### 🔧 Bug Fix: Extract Features Script Path Issue (2024-12-04)
**Fixed hardcoded path in `extract_features_and_save_pickle.py`**

**Issue**: The `pd_all_cells_mean.pickle` file (144MB) was being saved to the repository root instead of the designated tagged folder due to a hardcoded path in line 461.

**Fix**: Updated `write_pkl(pd_all_cells_mean, "pd_all_cells_mean")` to `write_pkl(pd_all_cells_mean, outpath)` to use the proper output directory path.

**Result**: All analysis files now correctly save to `data/pickle_files/extract_features/pickle_files_from_analysis/` maintaining the organized tagged folder structure.

## Project Organization

### Data Structure
```
data/
├── cells_min_30mins_long/          # Raw ABF files organized by cell
├── images/                         # All microscopy images, screenshots, etc.
├── illustations/                   # Figure illustrations and Affinity Designer files
├── pickle_files/                   # Processed data files
│   ├── analysis/                   # Analysis output pickle files
│   ├── all_data_with_training_df.pickle  # Main dataset
│   └── all_cell_firing_traces.pickle     # Firing traces
└── hdf5_files/                     # HDF5 format cell data
    └── cell_stats.h5               # Cell statistics
```

### Data Processing Workflow
1. **Raw ABF Files → HDF5 Conversion**: 
   - `conversion_scripts/convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py`

2. **HDF5 → Pickle Aggregation**: 
   - `conversion_scripts/compile_cell_dfs_to_pickle.py`

3. **Feature Extraction & Analysis**: 
   - `analysis_scripts/extract_features_and_save_pickle.py`

4. **Figure Generation**: 
   - **NEW**: Use `run_plotting_scripts.py` for automated figure generation
   - `plotting_scripts/main_figures/` - Main manuscript figures (Figures 1-6 only)
   - `plotting_scripts/supplementary_figures/` - All other figures (f_norm, supplementary, etc.)
   - `plotting_scripts/shared_utils/` - Shared plotting utilities

## Quick Start

### Running Analysis and Conversion Scripts

Use the new wrapper script for all data processing:

```bash
# List all available scripts and workflows
python run_analysis_conversion.py --list

# Run complete conversion pipeline (ABF → HDF5 → Pickle)
python run_analysis_conversion.py --workflow full_conversion

# Run analysis pipeline
python run_analysis_conversion.py --workflow full_analysis

# Run complete pipeline from raw data to analysis
python run_analysis_conversion.py --workflow complete_pipeline

# Run individual scripts
python run_analysis_conversion.py --script abf_to_hdf5
python run_analysis_conversion.py --script extract_features
```

### Running Plotting Scripts

Use the plotting script runner for figure generation:

```bash
# List all available figures
python run_plotting_scripts.py --list

# Run all main figures (standard analysis)
python run_plotting_scripts.py --figures all_main

# Run all supplementary figures
python run_plotting_scripts.py --figures all_supplementary

# Run specific figures
python run_plotting_scripts.py --figures figure_1 figure_2

# Run with field normalization
python run_plotting_scripts.py --figures figure_1_fnorm figure_2_fnorm
```

## Latest Version Management

**🎯 All scripts now use only the latest versions from main branch:**

### Version Control Summary
- **Figure 6**: Uses `figure_generation_script_6_v2.py` (latest, 2024-11-28)
- **Figure 7 Learner Comparison**: Uses `figure_generation_script_7_learner_non_learner_comparison_v3.py` (latest, 2024-12-04)
- **All other figures**: Use the most recent versions from main branch

### Script Organization
- **Main Figures**: 6 scripts (Figures 1-6, latest versions only)
- **Supplementary Figures**: 14 scripts (f_norm variants, supplementary figures)
- **Not Used**: Older versions moved to `plotting_scripts/not_used/`

### Benefits
- ✅ **No Version Confusion**: Only one version per figure type
- ✅ **Exact Main Branch Match**: Results identical to main branch
- ✅ **Clean Configuration**: `plotting_config.yaml` references latest versions only
- ✅ **Maintained Functionality**: All plotting capabilities preserved

See `LATEST_VERSIONS_CLEANUP_SUMMARY.md` for detailed cleanup documentation.

## Detailed Usage

### Data Organization
- **Images**: All `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` files are in `data/images/`
- **Illustrations**: All `.afdesign` files and figure illustrations are in `data/illustations/`
- **Pickle Files**: All analysis outputs are organized in `data/pickle_files/analysis/`
- **HDF5 Files**: Cell statistics and converted data in `data/hdf5_files/`

### Script Configuration
- **Plotting**: Configure in `plotting_config.yaml`
- **Analysis/Conversion**: Configure in `analysis_config.yaml`

### Workflows Available
1. **full_conversion**: ABF → HDF5 → Pickle conversion
2. **full_analysis**: Feature extraction and cell classification
3. **complete_pipeline**: Full pipeline from raw data to analysis

## File Structure

### Core Scripts
- `run_analysis_conversion.py` - Wrapper for analysis and conversion scripts
- `run_plotting_scripts.py` - Wrapper for plotting scripts
- `plotting_config.yaml` - Plotting configuration
- `analysis_config.yaml` - Analysis and conversion configuration

### Analysis Scripts
- `analysis_scripts/extract_features_and_save_pickle.py` - Main analysis pipeline
- `analysis_scripts/calculate_scale_bar_40x_automated_image_save.py` - Scale bar calculation

### Conversion Scripts
- `conversion_scripts/convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py` - ABF to HDF5
- `conversion_scripts/compile_cell_dfs_to_pickle.py` - HDF5 to pickle compilation
- `conversion_scripts/compile_firing_data_protocols_to_pickle_and_extract_features.py` - Firing data compilation

### Plotting Scripts
- `plotting_scripts/main_figures/` - Main manuscript figures (1-6)
- `plotting_scripts/supplementary_figures/` - Supplementary and f_norm figures
- `plotting_scripts/shared_utils/` - Shared plotting utilities
- `plotting_scripts/not_used/` - Deprecated scripts

## Requirements
- Python 3.8+
- See `requirements.txt` for package dependencies
- Recommended: 32GB RAM for full analysis pipeline

## Author
- **Anzal KS** (anzal.ks@gmail.com)
- Repository: https://github.com/anzalks/

## Notes
- All scripts maintain exact plot appearance from main branch
- Shared utilities ensure consistent plotting across all figures
- Wrapper scripts provide unified interface for all operations
- Data is organized by type for easy access and management


