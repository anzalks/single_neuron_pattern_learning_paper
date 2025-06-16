# Universal Subplot Labeling System

**Author:** Anzal KS (anzal.ks@gmail.com)  
**Repository:** https://github.com/anzalks/

## Overview

This document describes the implementation of a **universal subplot labeling system** for all figure generation scripts in the pattern learning paper. The system provides **consistent labeling methods** while preserving **exact text positions and all visual features**.

## Problem Solved

### Previous Issues:
- **4 different labeling methods** used across scripts
- Inconsistent function calls and implementations
- Difficult to maintain and modify
- No standardized approach

### Previous Methods:
1. **Direct text() calls:** `ax.text(0.05, 1.1, 'A', transform=ax.transAxes, ...)`
2. **label_axis() function:** `label_axis(axes_list, "H")` → generates "Hi", "Hii", "Hiii"
3. **Array-based labels:** `panel_labels = ['A', 'B', 'C']`
4. **Sequential assignment:** Individual text() calls for A through M

## New Universal System

### Core Functions (Added to `shared_utils/baisic_plot_fuctnions_and_features.py`)

#### 1. `add_subplot_label()` - Universal Single Label Function
```python
bpf.add_subplot_label(ax, label_text, xpos=-0.1, ypos=1.1, 
                     fontsize=16, fontweight='bold', ha='center', va='center')
```

**Features:**
- Handles all single subplot labels
- Preserves exact positioning parameters
- Consistent API across all scripts
- Support for all matplotlib text parameters

#### 2. `add_subplot_labels_from_list()` - Multiple Labels Function
```python
bpf.add_subplot_labels_from_list(axes_list, labels_list, 
                                base_params={'xpos': -0.1, 'ypos': 1.1, 
                                           'fontsize': 16, 'fontweight': 'bold'})
```

**Features:**
- Handles multiple axes with consistent parameters
- Individual parameter override capability
- Replaces label_axis() functionality

#### 3. `generate_letter_roman_labels()` - Roman Numeral Generator
```python
labels = bpf.generate_letter_roman_labels('H', 3)  # Returns ['Hi', 'Hii', 'Hiii']
```

#### 4. `int_to_roman()` - Roman Numeral Conversion
```python
roman = bpf.int_to_roman(3)  # Returns 'iii'
```

## Migration Examples

### Before & After Comparisons

#### Example 1: Direct Text Calls
**Before:**
```python
axs_img.text(0.05, 1.1, 'A', transform=axs_img.transAxes, 
             fontsize=16, fontweight='bold', ha='center', va='center')
```

**After:**
```python
bpf.add_subplot_label(axs_img, 'A', xpos=0.05, ypos=1.1, 
                     fontsize=16, fontweight='bold')
```

#### Example 2: label_axis() Calls
**Before:**
```python
label_axis(axs_fl_list, "D", xpos=-0.1, ypos=1.1)
```

**After:**
```python
fl_labels = bpf.generate_letter_roman_labels("D", len(axs_fl_list))
bpf.add_subplot_labels_from_list(axs_fl_list, fl_labels, 
                                base_params={'xpos': -0.1, 'ypos': 1.1, 
                                           'fontsize': 16, 'fontweight': 'bold'})
```

#### Example 3: Sequential Letters
**Before:**
```python
axs_rmp.text(-0.1, 1.1, 'A', transform=axs_rmp.transAxes, fontsize=16, fontweight='bold')
axs_pre.text(-0.1, 1.1, 'B', transform=axs_pre.transAxes, fontsize=16, fontweight='bold')
axs_post.text(-0.1, 1.1, 'C', transform=axs_post.transAxes, fontsize=16, fontweight='bold')
```

**After:**
```python
bpf.add_subplot_label(axs_rmp, 'A', xpos=-0.1, ypos=1.1, fontsize=16, fontweight='bold')
bpf.add_subplot_label(axs_pre, 'B', xpos=-0.1, ypos=1.1, fontsize=16, fontweight='bold')
bpf.add_subplot_label(axs_post, 'C', xpos=-0.1, ypos=1.1, fontsize=16, fontweight='bold')
```

## Automatic Migration Script

### Usage
```bash
python update_all_labeling.py
```

### What the Script Does:
1. **Creates backup** of all original scripts
2. **Automatically detects** and updates all 4 labeling patterns
3. **Preserves exact positioning** and formatting parameters
4. **Adds imports** if needed
5. **Provides detailed summary** of changes

### Script Features:
- **Regex-based pattern matching** for accurate detection
- **Backup creation** before any changes
- **Statistics tracking** for all updates
- **Error handling** and reporting
- **Dry-run capability** for testing

## Label Distribution Across Figures

### Main Figures (6 total):
- **Figure 1:** A, B, Ci-Ciii, Di-Diii, Ei-Eii, F
- **Figure 2:** A, B, C, Di-Diii, E  
- **Figure 3:** A, B, C, D, E, F, G, Hi-Hii, I, J, K
- **Figure 4:** Ai-Aiii, Bi-Biii, C, Di-Diii, Ei-Eii
- **Figure 5:** A, Bi-Biii, Ci-Ciii, D, Ei-Eiii, Fi-Fii
- **Figure 6:** A, Bi-Biii, Ci-Ciii

### Supplementary Figures (16+ total):
- **Field-normalized versions:** figure_2_fnorm through figure_7_fnorm
- **Dedicated supplementary:** supp_1, supp_2 (with variants)
- **Special analyses:** CHR2 sensitisation (A-C), RMP distribution (A-F)

## Label Visibility Control 🏷️

### NEW FEATURE: Global Label On/Off Control

You can now **enable or disable all alphabetical subplot labels** across all figures using several methods:

#### Method 1: Command Line Flags (Recommended)
```bash
# Generate figures WITHOUT labels
python run_plotting_scripts.py --main_fig --alpha_labels_off
python run_plotting_scripts.py --all_fig --no_labels

# Generate figures WITH labels (default behavior)
python run_plotting_scripts.py --main_fig --alpha_labels_on
python run_plotting_scripts.py --all_fig
```

#### Method 2: Programmatic Control
```python
import sys
sys.path.insert(0, 'plotting_scripts')
from shared_utils import baisic_plot_fuctnions_and_features as bpf

# Turn off all labels globally
bpf.set_subplot_labels_enabled(False)

# Generate figures - no labels will appear
# ... your plotting code here ...

# Turn labels back on
bpf.set_subplot_labels_enabled(True)

# Check current state
if bpf.get_subplot_labels_enabled():
    print("Labels are enabled")

# Toggle labels
bpf.toggle_subplot_labels()
```

#### Method 3: Environment Variable
```bash
# Turn off labels via environment variable
export SUBPLOT_LABELS_ENABLED=False
python plotting_scripts/main_figures/figure_generation_script_1.py

# Turn on labels
export SUBPLOT_LABELS_ENABLED=True
python plotting_scripts/main_figures/figure_generation_script_1.py
```

#### Method 4: Force Show Override
```python
# Force labels to show regardless of global setting
bpf.add_subplot_label(ax, 'A', xpos=0.05, ypos=1.1, force_show=True)

# Force multiple labels to show
bpf.add_subplot_labels_from_list(axes_list, labels_list, 
                                base_params={...}, force_show=True)
```

## Benefits of the New System

### ✅ Consistency
- **Single API** for all labeling operations
- **Standardized function calls** across all scripts
- **Uniform parameter handling**

### ✅ Maintainability  
- **Central implementation** in shared_utils
- **Easy to modify** labeling behavior globally
- **Clear documentation** and examples

### ✅ Preservation
- **Exact positioning** maintained
- **All visual features** preserved
- **No changes** to plot appearance

### ✅ Flexibility
- **Support for single labels** (A, B, C)
- **Support for Roman combinations** (Ai, Aii, Aiii)
- **Parameter override** capability
- **Global visibility control** (NEW!)
- **Future extensibility**

### ✅ Control Options
- **Command line flags** for batch processing
- **Programmatic control** for custom scripts  
- **Environment variables** for system-wide control
- **Individual override** for special cases

## Testing and Validation

### Pre-Migration Checklist:
- [ ] Create backup using migration script
- [ ] Review current labeling patterns in key figures
- [ ] Identify any custom labeling approaches

### Post-Migration Checklist:
- [ ] Run migration script
- [ ] Test generate one figure from each category:
  - [ ] Main figure (e.g., Figure 1)
  - [ ] Supplementary figure (e.g., Figure 3 fnorm)
  - [ ] Special analysis (e.g., RMP distribution)
- [ ] Verify all labels appear in correct positions
- [ ] Check that no visual changes occurred
- [ ] Confirm all scripts run without errors

### Validation Commands:
```bash
# Test main figures
python plotting_scripts/main_figures/figure_generation_script_1.py --outdir-path outputs/test

# Test supplementary figures  
python plotting_scripts/supplementary_figures/figure_generation_script_3_fnorm.py --outdir-path outputs/test

# Test special analyses
python plotting_scripts/supplementary_figures/figure_generation_script_rmp_distribution.py --outdir-path outputs/test

# Test label control functionality
python run_plotting_scripts.py --figures figure_1 --alpha_labels_off --output_dir outputs/test_no_labels
python run_plotting_scripts.py --figures figure_1 --alpha_labels_on --output_dir outputs/test_with_labels

# Compare outputs to verify label visibility control works
```

## Troubleshooting

### Common Issues:

#### Import Errors
**Problem:** `ModuleNotFoundError: No module named 'shared_utils'`
**Solution:** Ensure PYTHONPATH includes plotting_scripts directory

#### Missing Functions
**Problem:** `AttributeError: module 'baisic_plot_fuctnions_and_features' has no attribute 'add_subplot_label'`
**Solution:** Update shared_utils/baisic_plot_fuctnions_and_features.py with new functions

#### Position Changes
**Problem:** Labels appear in wrong positions
**Solution:** Check that xpos/ypos parameters exactly match original text() calls

### Debugging Tips:
1. **Compare with backup** files to identify differences
2. **Check function signatures** match expected parameters
3. **Verify import statements** are correct
4. **Test with single figure** before updating all

## Future Enhancements

### Potential Improvements:
- **Automatic position optimization** based on subplot size
- **Style themes** for different journal requirements  
- **Batch label updates** for figure revisions
- **Export label mapping** for figure legends
- **Integration with figure generation pipeline**

## File Structure

```
plotting_scripts/
├── shared_utils/
│   ├── baisic_plot_fuctnions_and_features.py  # ← New functions added here
│   ├── calcBoot2.py
│   └── __init__.py
├── main_figures/
│   ├── figure_generation_script_1.py  # ← Updated
│   ├── figure_generation_script_2.py  # ← Updated
│   └── ...
└── supplementary_figures/
    ├── figure_generation_script_2_fnorm.py  # ← Updated
    └── ...

backup_original_scripts/  # ← Created by migration script
└── plotting_scripts/     # ← Complete backup of originals

update_all_labeling.py    # ← Migration script
LABELING_SYSTEM_README.md # ← This documentation
```

## Support

For questions or issues with the labeling system:
- **Author:** Anzal KS
- **Email:** anzal.ks@gmail.com  
- **Repository:** https://github.com/anzalks/

---

**Note:** Always test the updated scripts thoroughly before using them for final figure generation. The migration preserves all visual aspects, but verification is recommended for critical publication figures. 