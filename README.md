# Single Neuron Pattern Learning Paper Analysis Pipeline

All scripts associated with the pattern learning paper - comprehensive analysis, conversion, and plotting pipeline for single neuron electrophysiology data.

## Table of Contents
- [Quick Start](#quick-start)
- [Usage Scenarios](#usage-scenarios)
- [Unified Figure Saving System](#unified-figure-saving-system)
- [Complete Project Structure](#complete-project-structure)
- [Data Processing Workflow](#data-processing-workflow)
- [Analysis Scripts Usage](#analysis-scripts-usage)
- [Plotting Scripts Usage](#plotting-scripts-usage)
- [Output Organization](#output-organization)
- [Statistical Tests Implementation](#statistical-tests-implementation)
- [Configuration Files](#configuration-files)
- [Requirements](#requirements)

## Quick Start

### 🚀 Complete Pipeline (From Raw Data to Figures)

```bash
# 1. Run complete analysis pipeline (ABF → HDF5 → Pickle → Feature Extraction)
python run_analysis_conversion.py --workflow complete_pipeline

# 2. Generate all main manuscript figures
python run_plotting_scripts.py --main_fig

# 3. Generate all supplementary figures
python run_plotting_scripts.py --supplementary_fig
```

## Usage Scenarios

### 📋 **Complete Workflows**

```bash
# 🎯 SCENARIO 1: Complete Pipeline (Raw Data → Analysis → All Figures)
python run_analysis_conversion.py --workflow complete_pipeline
python run_plotting_scripts.py --all_fig

# 🎯 SCENARIO 2: Analysis Only (if data already converted)
python run_analysis_conversion.py --workflow full_analysis
python run_plotting_scripts.py --all_fig

# 🎯 SCENARIO 3: Data Conversion Only
python run_analysis_conversion.py --workflow full_conversion
```

### 📊 **Plotting Scenarios**

```bash
# 🎨 Generate ALL figures (main + supplementary)
python run_plotting_scripts.py --all_fig

# 📄 Generate ONLY main manuscript figures (Figures 1-6)
python run_plotting_scripts.py --main_fig

# 📚 Generate ONLY supplementary figures
python run_plotting_scripts.py --supplementary_fig

# 🔬 Generate specific figures
python run_plotting_scripts.py --figures figure_1 figure_2 figure_3

# 📐 Generate field-normalized figures only
python run_plotting_scripts.py --analysis_type field_normalized
```

### 🔧 **Analysis Scenarios**

```bash
# 🔄 Complete data conversion pipeline
python run_analysis_conversion.py --workflow full_conversion

# 📊 Complete analysis pipeline (assumes data already converted)
python run_analysis_conversion.py --workflow full_analysis

# 🧪 Run individual analysis steps
python run_analysis_conversion.py --script extract_features
python run_analysis_conversion.py --script chr2_sensitisation

# 💾 Run individual conversion steps
python run_analysis_conversion.py --script abf_to_hdf5
python run_analysis_conversion.py --script compile_cells_to_pickle
```

### 📝 **Available Commands Reference**

**Analysis/Conversion Script Commands:**
```bash
# List all available options
python run_analysis_conversion.py --list

# Command structure
python run_analysis_conversion.py [--script SCRIPT_NAME | --workflow WORKFLOW_NAME] [--args "custom_args"]

# Available workflows:
# - complete_pipeline: Full pipeline (ABF → HDF5 → pickle → features → CHR2)
# - full_conversion: Data conversion only (ABF → HDF5 → pickle)  
# - full_analysis: Analysis only (features → CHR2)

# Available scripts:
# - abf_to_hdf5: Convert ABF files to HDF5
# - compile_cells_to_pickle: Combine HDF5 files to pickle
# - compile_firing_data: Extract firing properties
# - extract_features: Main feature extraction and cell classification
# - calculate_scale_bar: Process microscopy scale images
# - chr2_sensitisation: Generate CHR2 sensitisation data
```

**Plotting Script Commands:**
```bash
# List all available figures
python run_plotting_scripts.py --list

# Command structure  
python run_plotting_scripts.py [--all_fig | --main_fig | --supplementary_fig | --figures FIGURE_LIST] [--analysis_type TYPE]

# Bulk options:
# - --all_fig: Generate all figures (main + supplementary)
# - --main_fig: Generate main figures only (1-6)
# - --supplementary_fig: Generate supplementary figures only

# Specific options:
# - --figures: Space-separated list of specific figures
# - --analysis_type: 'standard' or 'field_normalized'
# - --output_dir: Custom output directory
# - --config: Custom config file path
```

### 📖 **Available Figures Reference**

| **Figure Type** | **Figure ID** | **Description** | **Command Example** |
|-----------------|---------------|-----------------|---------------------|
| **Main Figures** | | | |
| | `figure_1` | Overview | `python run_plotting_scripts.py --figures figure_1` |
| | `figure_2` | EPSP Analysis (★ 3 statistical tests) | `python run_plotting_scripts.py --figures figure_2` |
| | `figure_3` | Pattern Learning | `python run_plotting_scripts.py --figures figure_3` |
| | `figure_4` | Temporal Analysis | `python run_plotting_scripts.py --figures figure_4` |
| | `figure_5` | Plasticity | `python run_plotting_scripts.py --figures figure_5` |
| | `figure_6` | Network Analysis | `python run_plotting_scripts.py --figures figure_6` |
| **Supplementary** | | | |
| | `figure_2_fnorm` | EPSP Analysis - Field Normalized (★ 3 statistical tests) | `python run_plotting_scripts.py --figures figure_2_fnorm` |
| | `figure_3_fnorm` | Pattern Learning - Field Normalized | `python run_plotting_scripts.py --figures figure_3_fnorm` |
| | `figure_4_fnorm` | Temporal Analysis - Field Normalized | `python run_plotting_scripts.py --figures figure_4_fnorm` |
| | `figure_5_fnorm` | Plasticity - Field Normalized | `python run_plotting_scripts.py --figures figure_5_fnorm` |
| | `figure_6_v2_fnorm` | Network Analysis - Field Normalized | `python run_plotting_scripts.py --figures figure_6_v2_fnorm` |
| | `figure_7_v3` | Learner vs Non-Learner Comparison | `python run_plotting_scripts.py --figures figure_7_v3` |
| | `figure_7_fnorm` | Learner Comparison - Field Normalized | `python run_plotting_scripts.py --figures figure_7_fnorm` |
| | `supp_1` | Supplementary Figure 1 | `python run_plotting_scripts.py --figures supp_1` |
| | `supp_1_fnorm` | Supplementary Figure 1 - Field Normalized | `python run_plotting_scripts.py --figures supp_1_fnorm` |
| | `supp_2` | Supplementary Figure 2 | `python run_plotting_scripts.py --figures supp_2` |
| | `supp_2_fnorm` | Supplementary Figure 2 - Field Normalized | `python run_plotting_scripts.py --figures supp_2_fnorm` |
| | `supp_2_field_norm` | Supplementary Figure 2 - Field Norm | `python run_plotting_scripts.py --figures supp_2_field_norm` |
| | `supp_2_field_norm_fnorm` | Supplementary Figure 2 - Field Norm + fnorm | `python run_plotting_scripts.py --figures supp_2_field_norm_fnorm` |
| | `supp_chr2_sensitisation` | CHR2 Sensitisation Analysis | `python run_plotting_scripts.py --figures supp_chr2_sensitisation` |
| | `supp_rmp_distribution` | Resting Membrane Potential Distribution & Correlation Analysis (6 panels: A-F) | `python run_plotting_scripts.py --figures supp_rmp_distribution` |

**★ Note:** Figure 2 and Figure 2 fnorm automatically generate multiple statistical test versions (Wilcoxon, ANOVA, Mixed Effect Model)

### 💡 **Common Use Cases Examples**

```bash
# 📋 For new users - Complete pipeline from scratch
python run_analysis_conversion.py --workflow complete_pipeline
python run_plotting_scripts.py --all_fig

# 📊 Generate only main paper figures (fast)
python run_plotting_scripts.py --main_fig

# 🔬 Generate only statistical analysis figures (Figure 2 variants)
python run_plotting_scripts.py --figures figure_2 figure_2_fnorm

# 📚 Generate all supplementary figures for appendix
python run_plotting_scripts.py --supplementary_fig

# 🎯 Generate specific figure combinations
python run_plotting_scripts.py --figures figure_1 figure_3 figure_5 supp_1

# 🔄 Re-run analysis if data changed
python run_analysis_conversion.py --workflow full_analysis
python run_plotting_scripts.py --main_fig

# 💾 Only convert data (if you have new ABF files)
python run_analysis_conversion.py --workflow full_conversion
```

## Unified Figure Saving System

### 🎯 **Overview**

A comprehensive, production-ready system for consistent figure generation across all plotting scripts with global format control, quality settings, automatic filename management, and intelligent filename tagging.

### ✨ **Key Features**

- **🎨 Multiple Format Support**: PNG, PDF, SVG, EPS with optimized quality settings
- **🔧 Global Control**: Command-line flags and environment variables for scriptable automation  
- **📊 Quality Presets**: Standard (300 DPI) and high-quality (600 DPI) modes
- **🏷️ Smart Labeling**: Automatic subplot label control across all figures
- **🏷️ Filename Tagging**: Intelligent filename tagging based on label state (`_no_label` suffix)
- **📁 Auto Directory Creation**: Automatic output directory structure management
- **🔄 Backward Compatibility**: Seamless integration with existing scripts
- **⚡ Batch Processing**: Generate all figures simultaneously with consistent settings

### 🔧 **Command Line Flags**

| Flag | Description | Example |
|------|-------------|---------|
| `--format` | Single output format | `--format pdf` |
| `--multi_format` | Multiple output formats | `--multi_format png pdf svg` |
| `--dpi` | Custom DPI setting | `--dpi 450` |
| `--high_quality` | High quality mode (600 DPI) | `--high_quality` |
| `--transparent` | Transparent background | `--transparent` |
| `--alpha_labels_on` | Enable subplot labels (default) | `--alpha_labels_on` |
| `--alpha_labels_off` | Disable subplot labels | `--alpha_labels_off` |
| `--no_labels` | Alias for --alpha_labels_off | `--no_labels` |
| `--label_tag_on` | Enable filename tagging (default) | `--label_tag_on` |
| `--label_tag_off` | Disable filename tagging | `--label_tag_off` |

### 🏷️ **Filename Tagging System**

The system automatically adds intelligent tags to filenames based on the label state:

**🎯 Tagging Behavior:**
- **Labels ON**: `figure_1.png` (base filename)
- **Labels OFF**: `figure_1_no_label.png` (tagged filename)
- **Tagging disabled**: `figure_1.png` (always base filename)

**📋 Examples:**

```bash
# Generate figures with labels (default filename)
python run_plotting_scripts.py --figures figure_1 --alpha_labels_on --format png
# Output: figure_1.png

# Generate figures without labels (tagged filename)  
python run_plotting_scripts.py --figures figure_1 --alpha_labels_off --format png
# Output: figure_1_no_label.png

# Generate without labels but disable tagging (base filename)
python run_plotting_scripts.py --figures figure_1 --alpha_labels_off --label_tag_off --format png
# Output: figure_1.png

# Generate both versions with multiple formats
python run_plotting_scripts.py --figures figure_1 --alpha_labels_on --multi_format png pdf
python run_plotting_scripts.py --figures figure_1 --alpha_labels_off --multi_format png pdf
# Output: figure_1.png, figure_1.pdf, figure_1_no_label.png, figure_1_no_label.pdf
```

### 🎨 **Format Control Examples**

```bash
# Single format examples
python run_plotting_scripts.py --all_fig --format png
python run_plotting_scripts.py --all_fig --format pdf
python run_plotting_scripts.py --all_fig --format svg

# Multiple format examples
python run_plotting_scripts.py --all_fig --multi_format png pdf
python run_plotting_scripts.py --all_fig --multi_format png pdf svg eps

# Quality control examples
python run_plotting_scripts.py --all_fig --format png --dpi 600
python run_plotting_scripts.py --all_fig --high_quality --format pdf
python run_plotting_scripts.py --all_fig --transparent --format png

# Complete workflow examples
python run_plotting_scripts.py --all_fig --alpha_labels_on --multi_format png pdf
python run_plotting_scripts.py --all_fig --alpha_labels_off --multi_format png pdf
# This generates both labeled and unlabeled versions in PNG and PDF formats
```

### 🌍 **Environment Variables**

| Variable | Description | Example |
|----------|-------------|---------|
| `FIGURE_FORMAT` | Single format | `png`, `pdf`, `svg`, `eps` |
| `FIGURE_FORMATS` | Multiple formats | `png,pdf,svg` |
| `FIGURE_DPI` | DPI setting | `300`, `600` |
| `FIGURE_TRANSPARENT` | Transparency | `True`, `False` |
| `SUBPLOT_LABELS_ENABLED` | Subplot labels state | `True`, `False` |
| `FIGURE_LABEL_TAG` | Filename tagging | `True`, `False` |

**📋 Environment Variable Examples:**

```bash
# Set environment variables for batch processing
export FIGURE_FORMAT=pdf
export FIGURE_DPI=600
export SUBPLOT_LABELS_ENABLED=False
export FIGURE_LABEL_TAG=True

# Run with environment settings
python run_plotting_scripts.py --all_fig
# All figures will be generated as high-quality PDFs without labels, with _no_label tags

# Override environment with command line
python run_plotting_scripts.py --all_fig --format png --alpha_labels_on
# Overrides environment settings
```

### 📊 **Quality Settings**

| Format | Standard DPI | High Quality DPI | Transparency Support |
|--------|--------------|------------------|---------------------|
| PNG | 300 | 600 | ✅ Yes |
| PDF | 300 | 600 | ✅ Yes |
| SVG | Vector | Vector | ✅ Yes |
| EPS | Vector | Vector | ❌ No |

### 🔧 **Advanced Usage**

```bash
# Generate publication-ready figures (high quality, multiple formats)
python run_plotting_scripts.py --main_fig --high_quality --multi_format png pdf svg

# Generate presentation figures (transparent background)
python run_plotting_scripts.py --main_fig --transparent --format png

# Generate both labeled and unlabeled versions for all figures
python run_plotting_scripts.py --all_fig --alpha_labels_on --multi_format png pdf
python run_plotting_scripts.py --all_fig --alpha_labels_off --multi_format png pdf

# Custom DPI for specific requirements
python run_plotting_scripts.py --figures figure_1 figure_2 --dpi 450 --format png

# Batch processing with environment control
FIGURE_FORMATS=png,pdf,svg FIGURE_DPI=600 python run_plotting_scripts.py --all_fig
```

### 🎯 **Migration Status**

**✅ Complete Migration**: All 21 plotting scripts now use the unified saving system:
- **Main figures**: 6 scripts (figure_1 through figure_6_v2)  
- **Supplementary figures**: 15 scripts (all supplementary variants)
- **Total coverage**: 21/21 scripts (100% migrated)

**🔧 System Benefits:**
- ✅ **Consistent quality** across all figures (300 DPI default, 600 DPI high-quality)
- ✅ **Global format control** via command line or environment variables
- ✅ **Intelligent filename tagging** based on label state
- ✅ **Multi-format generation** in single command
- ✅ **Automatic directory creation** and organization
- ✅ **Backward compatibility** with existing workflows

## Complete Project Structure

```
single_neuron_pattern_learning_paper/
├── 📁 Root Configuration Files
│   ├── README.md                        # This comprehensive guide
│   ├── requirements.txt                 # Python dependencies
│   ├── LICENSE                         # AGPL v3 License
│   ├── .gitignore                      # Git ignore rules
│   ├── analysis_config.yaml            # Analysis pipeline configuration
│   ├── plotting_config.yaml            # Plotting scripts configuration
│   ├── run_analysis_conversion.py      # Main analysis/conversion runner
│   └── run_plotting_scripts.py         # Main plotting runner
│
├── 📁 conversion_scripts/              # Data format conversion scripts
│   ├── convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py
│   ├── compile_cell_dfs_to_pickle.py   # HDF5 → Pickle compilation
│   └── compile_firing_data_protocols_to_pickle_and_extract_features.py
│
├── 📁 analysis_scripts/                # Data analysis and feature extraction
│   ├── extract_features_and_save_pickle.py           # Main analysis pipeline
│   ├── calculate_scale_bar_40x_automated_image_save.py # Scale bar calculation
│   └── sensitisation_data_pickle_generation_from_training_data.py
│
├── 📁 plotting_scripts/               # Figure generation scripts
│   ├── 📁 main_figures/              # Main manuscript figures (1-6)
│   │   ├── figure_generation_script_1.py            # Figure 1: Overview
│   │   ├── figure_generation_script_2.py            # Figure 2: EPSP Analysis (3 statistical tests)
│   │   ├── figure_generation_script_3.py            # Figure 3: Pattern Learning
│   │   ├── figure_generation_script_4.py            # Figure 4: Temporal Analysis
│   │   ├── figure_generation_script_5.py            # Figure 5: Plasticity
│   │   └── figure_generation_script_6_v2.py         # Figure 6: Network Analysis
│   ├── 📁 supplementary_figures/     # Supplementary and field-normalized figures
│   │   ├── figure_generation_script_2_fnorm.py      # Figure 2 field-normalized (3 statistical tests)
│   │   ├── figure_generation_script_3_fnorm.py      # Figure 3 field-normalized
│   │   ├── figure_generation_script_4_fnorm.py      # Figure 4 field-normalized
│   │   ├── figure_generation_script_5_fnorm.py      # Figure 5 field-normalized
│   │   ├── figure_generation_script_6_v2_fnorm.py   # Figure 6 field-normalized
│   │   ├── figure_generation_script_7_learner_non_learner_comparison_v3.py  # Figure 7: Learner comparison
│   │   ├── figure_generation_script_7_fnorm.py      # Figure 7 field-normalized
│   │   ├── figure_generation_script_supp_1.py       # Supplementary Figure 1
│   │   ├── figure_generation_script_supp_1_fnorm.py # Supplementary Figure 1 field-normalized
│   │   ├── figure_generation_script_supp_2.py       # Supplementary Figure 2
│   │   ├── figure_generation_script_supp_2_fnorm.py # Supplementary Figure 2 field-normalized
│   │   ├── figure_generation_script_supp_2_field_norm.py      # Supplementary Figure 2 field norm
│   │   ├── figure_generation_script_supp_2_field_norm_fnorm.py # Supplementary Figure 2 field norm + fnorm
│   │   ├── Supplementary_figure_6_chr2_sensitiation.py        # CHR2 Sensitisation
│   │   └── figure_generation_script_rmp_distribution.py       # RMP Distribution & Correlation
│   └── 📁 shared_utils/              # Shared plotting utilities
│       └── baisic_plot_fuctnions_and_features.py    # Common plotting functions
│
├── 📁 data/                          # All experimental data (organized by type)
│   ├── 📁 abf_all_cells/            # Raw ABF files (by cell)
│   │   ├── 2022_12_12_cell_2/       # Individual cell directories
│   │   │   ├── *.abf                # Raw electrophysiology files (~32GB total)
│   │   │   └── result_plots_multi/  # Individual analysis plots
│   │   ├── 2022_12_12_cell_5/
│   │   ├── 2022_12_19_cell_2/
│   │   ├── ... (32 total cells)
│   │   └── 2023_03_22_cell_2/
│   ├── 📁 hdf5_files/               # Converted HDF5 data
│   │   └── abf_to_hdf5/
│   │       ├── all_cells_hdf/       # Individual cell HDF5 files (~15GB)
│   │       └── cell_stats.h5        # Combined cell statistics
│   ├── 📁 pickle_files/             # Processed analysis data
│   │   ├── compile_cells_to_pickle/ # Compiled cell data
│   │   │   └── all_data_df.pickle   # Master cell dataframe (~2GB)
│   │   ├── compile_firing_data/     # Firing properties
│   │   │   ├── all_cell_all_trial_firing_properties.pickle
│   │   │   └── all_cell_firing_traces.pickle
│   │   └── extract_features/        # Feature extraction outputs
│   │       └── pickle_files_from_analysis/
│   │           ├── all_cells_classified_dict.pickle       # Cell classification (120MB)
│   │           ├── all_cells_fnorm_classifeied_dict.pickle # Field-normalized classification (120MB)
│   │           ├── all_cells_inR.pickle                   # Input resistance data (77MB)
│   │           ├── baseline_traces_all_cells.pickle       # Baseline traces (1.2GB)
│   │           ├── pd_all_cells_all_trials.pickle         # All trials data (410MB)
│   │           ├── pd_all_cells_mean.pickle               # Mean responses (137MB)
│   │           ├── pd_training_data_all_cells_all_trials.pickle # Training data (8.4MB)
│   │           └── sensitisation_plot_data.pickle         # CHR2 sensitisation data (26MB)
│   ├── 📁 images/                   # Microscopy images and screenshots
│   │   ├── 40X1mm_micrometerslide_01mm_div.tiff          # Scale calibration
│   │   ├── with fluorescence and pipette.bmp             # Setup images
│   │   └── *.png, *.jpeg, *.bmp     # Various experimental images
│   └── 📁 illustations/            # Figure illustrations and design files
│       ├── figure_1.afdesign       # Affinity Designer source files
│       ├── figure_2_1.jpg          # Figure components
│       ├── figure_2_2.jpg
│       ├── figure_2_3.png
│       ├── Figure_3_1.png
│       ├── Figure_5_1.png
│       ├── Figure_6_1.png
│       └── *.afdesign              # Editable design files
│
└── 📁 outputs/                     # Generated figures (auto-created)
    ├── 📁 main_figures/            # Main manuscript figures
    │   ├── Figure_1/
    │   │   └── figure_1.png
    │   ├── Figure_2/               # ★ Multiple statistical test versions
    │   │   ├── figure_2_wilcox_sr_test.png        # Wilcoxon signed-rank
    │   │   ├── figure_2_rep_mesure_anova.png      # Repeated measures ANOVA
    │   │   ├── figure_2_mixd_effect_model.png     # Mixed effect model
    │   │   └── figure_2.png                      # Legacy version
    │   ├── Figure_3/
    │   ├── Figure_4/
    │   ├── Figure_5/
    │   └── Figure_6/
    └── 📁 supplementary_figures/   # Supplementary figures
        ├── Figure_2_fnorm/         # ★ Field-normalized with multiple stats
        │   ├── figure_2_fnorm_wilcox_sr_test.png
        │   ├── figure_2_fnorm_rep_mesure_anova.png
        │   ├── figure_2_fnorm_mixd_effect_model.png
        │   └── figure_2_fnorm.png
        ├── Figure_3_fnorm/
        ├── Figure_4_fnorm/
        ├── Figure_5_fnorm/
        ├── Figure_6_fnorm/
        ├── Figure_7/
        ├── Figure_7_fnorm/
        ├── Supplementary_figure_6_chr2_sensitisation/
        ├── Figure_RMP_Distribution/
        ├── supplimentary_figure_1/
        ├── supplimentary_figure_1_fnorm/
        ├── supplimentary_figure_2_field_norm/
        ├── supplimentary_figure_2_field_norm_fnorm/
        └── supplimentary_figure_2_fnorm/
```

## Data Processing Workflow

### 🔄 Complete Analysis Pipeline

```bash
# Option 1: Run entire pipeline at once
python run_analysis_conversion.py --workflow complete_pipeline

# Option 2: Run step by step
python run_analysis_conversion.py --workflow full_conversion    # ABF → HDF5 → Pickle
python run_analysis_conversion.py --workflow full_analysis      # Feature extraction
```

### 📋 Individual Pipeline Steps

```bash
# 1. Convert ABF files to HDF5 format
python run_analysis_conversion.py --script abf_to_hdf5

# 2. Compile HDF5 files to master pickle
python run_analysis_conversion.py --script compile_to_pickle

# 3. Extract firing properties
python run_analysis_conversion.py --script firing_data

# 4. Extract features and classify cells
python run_analysis_conversion.py --script extract_features

# 5. Generate sensitisation data
python run_analysis_conversion.py --script sensitisation_data
```

### 📊 Data Flow Overview

```
Raw ABF Files → HDF5 Files → Master Pickle → Feature Extraction → Classification → Analysis-Ready Data
     ↓              ↓            ↓               ↓                 ↓              ↓
  Individual     Combined    Structured      Extracted        Cell Types     Ready for
  recordings     format      dataframe       features       identified      plotting
     32GB         15GB          2GB            1.5GB           120MB         120MB
```

## Analysis Scripts Usage

### 🔬 extract_features_and_save_pickle.py

**Purpose**: Main analysis pipeline - extracts features, classifies cells, and generates analysis-ready datasets.

```bash
python run_analysis_conversion.py --script extract_features

# Or run directly:
python analysis_scripts/extract_features_and_save_pickle.py \
    --pickle-path data/pickle_files/compile_cells_to_pickle/all_data_df.pickle \
    --outdir data/pickle_files/extract_features/pickle_files_from_analysis/
```

**Outputs Generated**:
- `pd_all_cells_mean.pickle` (137MB) - Mean responses per cell/pattern/timepoint
- `all_cells_classified_dict.pickle` (120MB) - Standard cell classification
- `all_cells_fnorm_classifeied_dict.pickle` (120MB) - Field-normalized classification  
- `all_cells_inR.pickle` (77MB) - Input resistance measurements
- `baseline_traces_all_cells.pickle` (1.2GB) - Raw baseline traces
- `pd_training_data_all_cells_all_trials.pickle` (8.4MB) - Training protocol data

### 🔧 convert_abf_to_hdf5_cell_by_cell.py

**Purpose**: Converts raw ABF electrophysiology files to structured HDF5 format.

```bash
python run_analysis_conversion.py --script abf_to_hdf5

# Manual execution:
python conversion_scripts/convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py \
    --abf-dir data/abf_all_cells/ \
    --outdir data/hdf5_files/abf_to_hdf5/
```

### 📦 compile_cell_dfs_to_pickle.py

**Purpose**: Compiles individual HDF5 files into master pickle dataframe.

```bash
python run_analysis_conversion.py --script compile_to_pickle
```

## Plotting Scripts Usage

### 🎨 Main Figure Generation

```bash
# Generate all main figures (Figures 1-6)
python run_plotting_scripts.py --main_fig

# Generate specific main figures
python run_plotting_scripts.py --figures figure_1 figure_2 figure_3

# View available figures
python run_plotting_scripts.py --list
```

### 📈 Supplementary Figure Generation  

```bash
# Generate all supplementary figures
python run_plotting_scripts.py --supplementary_fig

# Generate field-normalized versions
python run_plotting_scripts.py --figures figure_2_fnorm figure_3_fnorm

# Generate specific supplementary figures
python run_plotting_scripts.py --figures supp_1 supp_2 supp_chr2_sensitisation supp_rmp_distribution
```

### ★ Advanced Statistical Analysis (Figure 2 & 2 fnorm)

**Figure 2 and Figure 2 fnorm automatically generate three versions with different statistical tests:**

```bash
# Generate Figure 2 with all statistical tests
python run_plotting_scripts.py --figures figure_2

# Outputs created:
# - figure_2_wilcox_sr_test.png      (Wilcoxon signed-rank test)
# - figure_2_rep_mesure_anova.png    (Repeated measures ANOVA)  
# - figure_2_mixd_effect_model.png   (Mixed effect model)

# Generate Figure 2 fnorm with all statistical tests
python run_plotting_scripts.py --figures figure_2_fnorm

# Outputs created:
# - figure_2_fnorm_wilcox_sr_test.png
# - figure_2_fnorm_rep_mesure_anova.png
# - figure_2_fnorm_mixd_effect_model.png
```

### 📊 Individual Script Execution

```bash
# Run individual plotting scripts (advanced users)
cd plotting_scripts

# Main figures
python main_figures/figure_generation_script_2.py \
    --pikl-path ../data/pickle_files/extract_features/pickle_files_from_analysis/pd_all_cells_mean.pickle \
    --sortedcell-path ../data/pickle_files/extract_features/pickle_files_from_analysis/all_cells_classified_dict.pickle \
    --outdir-path ../outputs/main_figures/

# Supplementary figures  
python supplementary_figures/figure_generation_script_2_fnorm.py \
    --pikl-path ../data/pickle_files/extract_features/pickle_files_from_analysis/pd_all_cells_mean.pickle \
    --sortedcell-path ../data/pickle_files/extract_features/pickle_files_from_analysis/all_cells_fnorm_classifeied_dict.pickle \
    --outdir-path ../outputs/supplementary_figures/
```

## Output Organization

### 📁 Figure Outputs Structure

All generated figures are saved to `outputs/` directory with organized subdirectories:

- **Main Figures**: `outputs/main_figures/Figure_X/`
- **Supplementary Figures**: `outputs/supplementary_figures/Figure_X_fnorm/`
- **File Naming**: Descriptive names indicating statistical method used

### 🔢 Statistical Test Versions

**Figure 2 (Main) & Figure 2 fnorm (Supplementary)** generate multiple versions:

1. **Wilcoxon Signed-Rank Test** (`*_wilcox_sr_test.png`)
   - Non-parametric paired test
   - Default method, robust to outliers
   
2. **Repeated Measures ANOVA** (`*_rep_mesure_anova.png`)
   - Parametric test with Bonferroni correction
   - Tests main effect with post-hoc comparisons
   
3. **Mixed Effect Model** (`*_mixd_effect_model.png`)
   - Advanced linear mixed model
   - Accounts for random effects, FDR correction

### 📊 File Size Reference

- **Individual Figures**: ~175KB each (PNG format)
- **Total Main Figures**: ~1MB (6 figures)
- **Total Supplementary**: ~3MB (17 figures)
- **Figure 2 variants**: ~700KB (4 versions including statistical tests)

## Statistical Tests Implementation

### 🧮 Technical Details

**Repeated Measures ANOVA**:
- Uses `pingouin.rm_anova()` with subject ID as repeated measure
- Post-hoc: Pairwise t-tests with Bonferroni correction
- Fallback: Individual paired t-tests if ANOVA fails

**Mixed Effect Model**:
- Uses `statsmodels.mixedlm()` with random intercept per subject
- Dummy coding for time points (pre = reference)
- Fallback: FDR-corrected paired t-tests

**Error Handling**:
- All methods include graceful fallbacks
- Console output shows which method was used
- Maintains visual consistency across all versions

### 📋 Data Structure for Statistical Tests

All tests compare **pre vs post timepoints** (post_0, post_1, post_2, post_3) across three stimulus patterns:
- **Pattern 0**: Trained pattern
- **Pattern 1**: Overlapping pattern  
- **Pattern 2**: Non-overlapping pattern

## Configuration Files

### ⚙️ plotting_config.yaml

Controls all figure generation parameters:

```yaml
figures:
  main_figures:
    figure_2:
      standard:
        script: "main_figures/figure_generation_script_2.py"
        args:
          file: "data/pickle_files/extract_features/pickle_files_from_analysis/pd_all_cells_mean.pickle"
          stats: "data/pickle_files/extract_features/pickle_files_from_analysis/all_cells_classified_dict.pickle"
          # ... other parameters
```

### ⚙️ analysis_config.yaml

Controls analysis pipeline parameters:

```yaml
workflows:
  complete_pipeline:
    description: "Full pipeline from ABF to analysis"
    scripts: ["abf_to_hdf5", "compile_to_pickle", "firing_data", "extract_features"]
```

## Requirements

### 💻 System Requirements

- **Python**: 3.8+ (tested with 3.9-3.11)
- **Memory**: 32GB RAM recommended for full pipeline
- **Storage**: ~50GB free space for complete dataset
- **OS**: macOS, Linux, Windows (tested on macOS)

### 📦 Python Dependencies

Install with: `pip install -r requirements.txt`

**Key Dependencies**:
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical plotting
- `scipy>=1.7.0` - Scientific computing
- `pingouin>=0.5.0` - Statistical analysis
- `statsmodels>=0.13.0` - Advanced statistics
- `h5py>=3.1.0` - HDF5 file handling
- `neo>=0.10.0` - Electrophysiology data handling
- `statannotations>=0.2.3` - Statistical plot annotations
- `pillow>=8.0.0` - Image processing
- `pyyaml>=6.0` - Configuration files
- `tqdm>=4.60.0` - Progress bars

### 🚀 Installation

```bash
# Clone repository
git clone https://github.com/anzalks/single_neuron_pattern_learning_paper.git
cd single_neuron_pattern_learning_paper

# Install dependencies
pip install -r requirements.txt

# Verify installation
python run_plotting_scripts.py --list
python run_analysis_conversion.py --list
```

## Latest Updates

### 🔧 Recent Improvements (2024-12-07)

**Multi-Statistical Test Implementation**:
- Added three statistical test methods for Figure 2 and Figure 2 fnorm
- Wilcoxon signed-rank test (original), Repeated measures ANOVA, Mixed effect model
- Automatic generation of all statistical variants with single command
- Graceful fallbacks and error handling for robust analysis

**Bug Fixes**:
- Fixed hardcoded path in `extract_features_and_save_pickle.py` (2024-12-04)
- All analysis files now correctly save to designated output directories
- Maintained organized tagged folder structure for all outputs

**Pipeline Improvements**:
- Unified plotting runner system with comprehensive configuration
- Version control cleanup - only latest versions maintained
- Enhanced documentation with detailed usage instructions

## Author & License

**Author**: Anzal KS  
**Email**: anzal.ks@gmail.com  
**Repository**: https://github.com/anzalks/single_neuron_pattern_learning_paper  
**License**: GNU Affero General Public License v3.0 (AGPL v3)

## Notes & Tips

### 💡 Best Practices

1. **Always run analysis from project root**: Ensures proper imports and paths
2. **Use wrapper scripts**: `run_plotting_scripts.py` and `run_analysis_conversion.py` handle all dependencies
3. **Check output directories**: Figures save to organized subdirectories automatically
4. **Monitor memory usage**: Large datasets require substantial RAM
5. **Backup analysis outputs**: Pickle files take hours to generate

### 🔧 Troubleshooting

**Import Errors**: Ensure you're running from project root directory
**Memory Errors**: Reduce batch size or use a machine with more RAM  
**File Not Found**: Check that data files exist in expected locations
**Statistical Test Failures**: Check console output for fallback method used

### 🔄 Version Control

This project uses Git for version control. All scripts maintain exact plot appearance and analysis consistency. The project follows semantic versioning for releases.

### 📊 Data Reproducibility

All analysis outputs are deterministic and reproducible given the same input data. Random seeds are fixed where applicable. Statistical tests include multiple methods for robustness validation.

### 🎯 Citing This Work

If you use this code or data in your research, please cite the associated publication and link to this repository: https://github.com/anzalks/single_neuron_pattern_learning_paper

---

**License Notice**: This project is licensed under AGPL v3. If you modify and distribute this software, you must also provide the source code of your modifications under the same license terms.