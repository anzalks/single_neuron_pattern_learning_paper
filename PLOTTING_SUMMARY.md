# Plotting Script Reorganization Summary

## What Was Accomplished

### 1. Directory Structure Reorganization ✅
- **`plotting_scripts/main_figures/`** - Contains exactly 6 scripts (Figures 1-6, standard analysis only)
- **`plotting_scripts/supplementary_figures/`** - Contains 16 scripts (all f_norm versions, figures 7-8, supplementary figures)
- **`plotting_scripts/shared_utils/`** - Contains shared `baisic_plot_fuctnions_and_features.py` module
- **`plotting_scripts/not_used/`** - Contains deprecated scripts

### 2. Import Path Updates ✅
All plotting scripts now use:
```python
from shared_utils import baisic_plot_fuctnions_and_features as bpf
```
Instead of:
```python
import baisic_plot_fuctnions_and_features as bpf
```

### 3. Shared Utilities Setup ✅
- Copied `baisic_plot_fuctnions_and_features.py` from main branch to `plotting_scripts/shared_utils/`
- Copied supporting modules (`calcBoot.py`, `calcBoot2.py`)
- Created proper `__init__.py` for the package

### 4. Automated Script Runner ✅
Created `run_plotting_scripts.py` with capabilities:
- List all available figures
- Run all main figures (standard analysis)
- Run all field normalized figures 
- Run specific figures by name
- Automatic argument mapping from `config.yaml`
- Proper Python path setup for imports

### 5. Configuration Management ✅
Updated `config.yaml` with:
- Complete figure definitions for main and supplementary figures
- Argument mappings for all scripts
- Data file paths
- Output directory structure

### 6. Documentation ✅
- Updated `README.md` with new automated workflow
- Created `PLOTTING_USAGE.md` with detailed usage examples
- All documentation reflects the new structure

## Key Benefits

1. **Script Integrity**: All scripts are exact copies from main branch - plots will look identical
2. **Clean Organization**: Clear separation between main figures (1-6) and supplementary figures
3. **Automated Workflow**: Single command can generate all figures of a specific type
4. **Shared Utilities**: Common plotting functions centralized in one location
5. **Flexible Usage**: Can run individual figures or batches as needed

## Usage Examples

### Quick Start
```bash
# List all available figures
python run_plotting_scripts.py --list

# Generate all main manuscript figures (1-6)
python run_plotting_scripts.py --analysis_type standard

# Generate all field normalized versions
python run_plotting_scripts.py --analysis_type field_normalized

# Generate specific figures
python run_plotting_scripts.py --figures figure_1 figure_3 figure_7_fnorm
```

### Directory Contents

**Main Figures (6 scripts):**
1. `figure_generation_script_1.py`
2. `figure_generation_script_2.py` 
3. `figure_generation_script_3.py`
4. `figure_generation_script_4.py`
5. `figure_generation_script_5.py`
6. `figure_generation_script_6_v2.py`

**Supplementary Figures (16 scripts):**
- 6 f_norm versions of main figures (2-6, plus 6_v2)
- 2 additional figures (7, 8) 
- 3 f_norm versions of additional figures
- 5 dedicated supplementary figures (supp_1, supp_2 variants)

## Verification

The import updates were successfully applied to all 29 plotting scripts:
- ✅ All scripts now import from `shared_utils`
- ✅ No duplicate import statements
- ✅ Maintains exact same functionality as main branch
- ✅ Scripts can find the shared utilities module

## Next Steps

The plotting system is now ready for use. Users can:
1. Run the automated script to generate all figures
2. Generate specific subsets based on analysis type
3. Run individual figures as needed
4. All outputs will match the main branch exactly while using the new organized structure 