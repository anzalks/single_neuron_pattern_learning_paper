# Conversion and Analysis Scripts - Workflows Summary

**Author:** Anzal (anzal.ks@gmail.com)  
**Repository:** https://github.com/anzalks/  
**Date:** December 2024

## 🧹 Cleanup Completed

### ✅ Scripts Cleaned Up:
- **calcBoot scripts**: Kept only `calcBoot2.py` (latest, 2024-11-27) in `shared_utils/`
- **Random scripts deleted**: Removed `src/`, `tests/`, and `curve_fit_scripts/` directories
- **Markdown files**: Removed all `.md` files except `README.md`
- **Test files**: Deleted `test_shared_utilities.py` and related test infrastructure

## 📋 Conversion Scripts Workflows

### 1. ABF to HDF5 Conversion
**Script:** `conversion_scripts/convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py`

#### **Input Requirements:**
- **Folder Structure Expected:**
  ```
  cells_path/
  ├── 230622_cell_1/          # Individual cell folders
  │   ├── *.abf               # ABF files with specific naming
  │   └── ...
  ├── 230622_cell_2/
  └── ...
  ```

#### **File Detection Logic:**
- **ABF File Scanning**: Recursively finds all `*.abf` files using `glob('**/*abf')`
- **Protocol Detection**: Reads protocol names from ABF headers using `reader._axon_info['sProtocolPath']`
- **Training File Detection**: Searches for files with 'training' or 'Training' in protocol name
- **Pre/Post Sorting**: Automatically splits files into pre-training and post-training based on training file position

#### **Protocol Classification:**
The script automatically tags protocols based on filename patterns:
- **Points**: `'12_points'`, `'42_points'` → Spatial stimulation patterns
- **Patterns**: `'Baseline_5_T_1_1_3_3'`, `'patternsx'`, `'patterns_x'` → Temporal patterns  
- **Training**: `'Training'`, `'training'` → LTP induction protocols
- **RMP**: `'RMP'` → Resting membrane potential measurements
- **Input Resistance**: `'Input_res'` → Cell resistance measurements
- **Current Steps**: `'threshold'` → Current injection protocols

#### **Channel Mapping:**
- **Cell Recording**: `'IN0'` channel → intracellular voltage
- **Field Recording**: `'Field'` channel → extracellular field potential
- **Photodiode**: `'Photodiode'` → light stimulation monitor
- **TTL**: `'FrameTTL'` → stimulation timing signals

#### **Output:**
- **HDF5 Files**: `{cell_ID}_with_training_data.h5` (one per cell)
- **Cell Statistics**: `cell_stats.h5` (summary of all cells)
- **Output Directory**: `hdf_format_files_with_training/`

#### **Processing Features:**
- **Multiprocessing**: Uses 6 parallel processes
- **TTL Detection**: Automatically finds stimulation onset times
- **Frame Classification**: Distinguishes between point, pattern, and training protocols
- **Health Assessment**: Evaluates cell viability during recording

---

### 2. HDF5 to Pickle Compilation  
**Script:** `conversion_scripts/compile_cell_dfs_to_pickle.py`

#### **Input Requirements:**
- **HDF5 Files**: Directory containing `*.h5` files from ABF conversion
- **Cell Statistics**: `cell_stats.h5` file for filtering valid cells

#### **Workflow:**
1. **File Discovery**: Scans directory for all `*.h5` files using `glob('**/*h5')`
2. **Cell Validation**: Filters cells based on `cell_stats['cell_status'] == 'valid'`
3. **Data Compilation**: Concatenates all valid cell data into single DataFrame
4. **Output**: `all_data_df.pickle` in `pickle_file_all_cells_trail_corrected/`

#### **Quality Control:**
- Automatically excludes cells marked as invalid in cell statistics
- Preserves cell identity and experimental conditions
- Creates unified dataset for downstream analysis

---

### 3. Firing Data Compilation
**Script:** `conversion_scripts/compile_firing_data_protocols_to_pickle_and_extract_features.py`

#### **Input Requirements:**
- **Cell Folders**: Individual cell directories containing ABF files
- **Cell Statistics**: HDF5 file to filter valid cells
- **Protocol Filter**: Only processes files with 'cell_threshold' in protocol name

#### **Specific Processing:**
- **Target Protocols**: Specifically looks for current injection protocols (`cell_threshold`)
- **Current Extraction**: Reads injected current values from protocol metadata
- **Spike Detection**: Uses `scipy.signal.find_peaks()` to detect action potentials
- **Feature Extraction**: Calculates spike frequency, injected current relationships

#### **Output Files:**
- **Raw Traces**: `all_cell_firing_traces.pickle` 
- **Spike Properties**: `all_cell_all_trial_firing_properties.pickle`
- **Output Directory**: `pickle_format_files_firing_rate_data/`

---

## 📊 Analysis Scripts Workflows

### 1. Main Feature Extraction and Analysis
**Script:** `analysis_scripts/extract_features_and_save_pickle.py`

#### **Input Requirements:**
- **Main Dataset**: `all_data_with_training_df.pickle` (giant consolidated file)
- **Cell Statistics**: `cell_stats.h5` for validation
- **Memory**: Configurable for 32GB RAM usage (`--use-full-ram`)

#### **Complete Analysis Pipeline:**

##### **Phase 1: Baseline Data Extraction**
- **Target**: Pre-training pattern responses only
- **Filter**: `pre_post_status == "pre"` AND `frame_status == "pattern"`
- **Output**: `baseline_traces_all_cells.pickle`

##### **Phase 2: Input Resistance Analysis** 
- **Target**: Current injection responses
- **Calculations**: Sag ratio, input resistance
- **Output**: `all_cells_inR.pickle`

##### **Phase 3: Trial-by-Trial Feature Extraction**
- **Multiprocessing**: Up to 12 cores for 32GB RAM mode
- **Features Extracted Per Trial:**
  - **EPSP amplitude**: Peak positive deflection
  - **IPSP amplitude**: Peak negative deflection  
  - **Rise time**: Slope calculation (3-10ms window)
  - **Field potential**: Extracellular field response
  - **Mini EPSPs**: Spontaneous events (bandpass filtered 10-1000Hz)
  - **Areas**: Positive, negative, absolute under curve
- **Output**: `pd_all_cells_all_trials.pickle`

##### **Phase 4: Mean Response Calculation**
- **Grouping**: By cell, protocol, time point
- **Statistics**: Mean ± SEM for each feature
- **Output**: `pd_all_cells_mean.pickle`

##### **Phase 5: Training Data Extraction**
- **Target**: All training protocol responses
- **Output**: `pd_training_data_all_cells_all_trials.pickle`

##### **Phase 6: Cell Classification (Standard)**
- **Criteria**: Post-training vs Pre-training amplitude comparison
- **Learners (AP)**: `post_3 > pre` amplitude  
- **Non-learners (AN)**: `post_3 < pre` amplitude
- **Threshold**: Minimum 0.5mV pre-training response
- **Output**: `all_cells_classified_dict.pickle`

##### **Phase 7: Field-Normalized Classification**
- **Normalization**: EPSP amplitude / field potential amplitude  
- **Same criteria as standard but with normalized values**
- **Output**: `all_cells_fnorm_classifeied_dict.pickle`

#### **Quality Control Features:**
- **Response Threshold**: Excludes cells with <0.5mV pre-training responses
- **Recording Duration**: Requires minimum 6 time points
- **Health Monitoring**: Tracks RMP stability throughout experiment

---

### 2. Scale Bar Calculation
**Script:** `analysis_scripts/calculate_scale_bar_40x_automated_image_save.py`

#### **Input Requirements:**
- **Calibration Image**: Micrometer slide image (TIFF format)
- **Physical Distance**: Known distance between calibration bars (10μm)

#### **Workflow:**
1. **Image Processing**: Converts to grayscale and inverts if needed
2. **Bar Detection**: Uses projection profiles and peak detection
3. **Scaling Calculation**: Pixels per micrometer ratio
4. **Scale Bar Addition**: Adds calibrated scale bars to images
5. **Output**: Annotated images with scale bars

#### **Usage**: Primarily for microscopy image calibration in figures

---

## 🔧 Key Technical Features

### **File Type Requirements:**
- **ABF Files**: Axon Binary Format from electrophysiology
- **HDF5 Files**: Hierarchical Data Format for structured storage  
- **Pickle Files**: Python binary serialization for DataFrames
- **Image Files**: TIFF/PNG for microscopy data

### **Directory Structure Expected:**
```
experiment_data/
├── cell_folders/           # Individual cell recordings
│   ├── {date}_cell_{n}/   # Named with date and cell number
│   └── *.abf              # Protocol files with descriptive names
├── hdf_format_files/      # Converted HDF5 data
├── pickle_files/          # Analysis outputs
└── images/               # Microscopy calibration images
```

### **Automatic Protocol Detection:**
All scripts use filename pattern matching to automatically:
- Identify protocol types (training, patterns, points, etc.)
- Sort temporal sequence (pre/post training)
- Extract experimental parameters
- Validate data completeness

### **Parallel Processing:**
- **ABF Conversion**: 6 processes for file I/O intensive operations
- **Feature Extraction**: Up to 12 processes for CPU intensive calculations
- **Memory Management**: Configurable for large dataset processing

---

## 🎯 Summary

These scripts form a complete pipeline from raw electrophysiology data to publication-ready analysis:

1. **ABF → HDF5**: Raw binary files → Structured data with metadata
2. **HDF5 → Pickle**: Multi-file dataset → Single consolidated DataFrame  
3. **Feature Extraction**: Raw traces → Quantified physiological parameters
4. **Classification**: Parameter analysis → Cell type identification
5. **Quality Control**: Automated filtering → Reliable dataset

The system automatically handles:
- **File discovery and organization**
- **Protocol identification and sorting** 
- **Quality assessment and filtering**
- **Feature extraction and statistical analysis**
- **Multi-format output for different analysis needs**

**Result**: Fully automated pipeline from raw recordings to classified cell populations ready for figure generation. 