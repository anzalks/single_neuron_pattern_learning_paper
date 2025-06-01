# Plotting Scripts Usage Guide

This document explains how to use the reorganized plotting scripts for the pattern learning paper.

## Organization

- **`plotting_scripts/main_figures/`** - Contains scripts for main manuscript figures 1-6 (standard analysis only)
- **`plotting_scripts/supplementary_figures/`** - Contains all other scripts (f_norm versions, figures 7-8, supplementary figures)
- **`plotting_scripts/shared_utils/`** - Contains the shared `baisic_plot_fuctnions_and_features.py` module
- **`plotting_scripts/not_used/`** - Contains deprecated scripts

## Running Scripts

Use the `run_plotting_scripts.py` script to run figures based on the configuration in `config.yaml`.

### List Available Figures

```bash
python run_plotting_scripts.py --list
```

### Run All Main Figures (Standard Analysis)

```bash
python run_plotting_scripts.py --analysis_type standard
```

### Run All Field Normalized Figures

```bash
python run_plotting_scripts.py --analysis_type field_normalized
```

### Run Specific Figures

```bash
# Run specific main figures
python run_plotting_scripts.py --figures figure_1 figure_2 figure_3

# Run specific supplementary figures
python run_plotting_scripts.py --figures figure_7 supp_1 figure_2_fnorm

# Mix of main and supplementary
python run_plotting_scripts.py --figures figure_1 figure_7_fnorm supp_2
```

### Specify Output Directory

```bash
python run_plotting_scripts.py --figures figure_1 --output_dir ./my_outputs
```

### Use Custom Config File

```bash
python run_plotting_scripts.py --config my_config.yaml --figures figure_1
```

## Examples

### Generate All Main Manuscript Figures
```bash
python run_plotting_scripts.py --analysis_type standard
```
This will run figures 1-6 and output to `outputs/main_figures/`

### Generate All Field Normalized Versions
```bash
python run_plotting_scripts.py --analysis_type field_normalized
```
This will run all *_fnorm scripts and output to `outputs/supplementary_figures/`

### Generate Specific Figure for Paper
```bash
# Generate Figure 3 for main manuscript
python run_plotting_scripts.py --figures figure_3

# Generate Figure 3 field normalized version for supplementary
python run_plotting_scripts.py --figures figure_3_fnorm
```

## Script Arguments

The runner automatically maps configuration arguments to script command-line flags:

- `file` → `-f` (main data file)
- `stats` → `-s` (statistics/classification file)
- `resistance` → `-r` (resistance data)
- `image` → `-i` (image file)
- `projimg` → `-p` (projection image)
- `all_trials` → `-t` (all trials data)
- `cell_stats` → `-c` (cell statistics)
- `firing` → `-q` (firing properties)

## Important Notes

1. **Scripts are unchanged**: All plotting scripts are exact copies from the main branch with no modifications
2. **Shared utilities**: The `baisic_plot_fuctnions_and_features.py` module is available via the `shared_utils` package
3. **Python path**: The runner automatically adds the plotting_scripts directory to PYTHONPATH so scripts can find shared_utils
4. **Output consistency**: All plots should look exactly the same as they did in the main branch

## Troubleshooting

### Import Errors
If scripts can't find the shared utilities, ensure:
- The `shared_utils` directory exists in `plotting_scripts/`
- The `__init__.py` file is present in `shared_utils/`
- You're running the script from the project root directory

### Missing Data Files
Ensure all data files referenced in `config.yaml` exist:
- Pickle files in `analysis_scripts/pickle_files_from_analysis/`
- Image files in `data/illustations/`
- Cell statistics in `data/pickle_files/`

### Script Failures
Check that:
- All required Python packages are installed (see `requirements.txt`)
- Data file paths in `config.yaml` are correct
- Output directories are writable 