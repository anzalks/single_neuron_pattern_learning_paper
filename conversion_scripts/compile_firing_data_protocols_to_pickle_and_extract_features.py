__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import neo.io as nio
import numpy as np
import pandas as pd
import time
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import scipy

class Args: pass 
args_ = Args()

def list_folder(p):
    f_list = []
    f_list = list(p.glob('*_cell_*'))
    f_list.sort()
    return f_list

def list_files(p):
    f_list = []
    f_list=list(p.glob('**/*abf'))
    f_list.sort()
    return f_list

def write_pkl(file_to_write,file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(file_to_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f" wrote file to pickle")

def protocol_file_name(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
    #print(f"protocol_name:{protocol_name}.....")
    return protocol_name

def current_injected(reader):
    protocol_raw = reader.read_raw_protocol()
    protocol_raw = protocol_raw[0]
    protocol_trace = []
    for n in protocol_raw:
        protocol_trace.append(n[0])
    #i_min = np.abs(np.min(protocol_trace))
    #i_max = np.abs(np.max(protocol_trace))
    #i_av = np.around((i_max-i_min),2)
    return protocol_trace#i_av

def abf_to_df(file_name):
    df_from_abf = []
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels = reader.header['signal_channels']
    chan_count = len(channels)
    file_id = file_name.stem
    block  = reader.read_block(signal_group_mode='split-all')
    segments = block.segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate.magnitude
    sampling_rate_unit = str(sample_trace.sampling_rate.units).split()[-1]
    ti = sample_trace.t_start
    tf = sample_trace.t_stop
    protocol_trace_list=current_injected(reader)
    #print(protocol_trace_list)
    for s, segment in enumerate(segments):
        df_segment = pd.DataFrame()                                                   
        cell_signal = segment.analogsignals[0]
        cell_Signal_unit = str(cell_signal.units).split()[-1]
        cell_trace = np.hstack(np.ravel(np.array(cell_signal)))                        
        t = np.linspace(0,float(tf-ti),len(cell_trace))
        trial_no = s
        trial_no = [trial_no]*len(t)
        df_segment['trial_no']=trial_no
        df_segment[f'cell_trace({cell_Signal_unit})']=cell_trace
        df_segment["injected_current(pA)"]=protocol_trace_list[s]
        df_segment['time(s)']=t
        df_from_abf.append(df_segment)
    df_from_abf = pd.concat(df_from_abf,ignore_index=True)
    df_from_abf.insert(loc=0, column=f'sampling_rate({sampling_rate_unit})', value=sampling_rate)
    return df_from_abf

def extract_first_spike(cell_data):
    trace = cell_data["cell_trace(mV)"].to_numpy()
    peaks, properties = scipy.signal.find_peaks(trace, height=0)
    first_spike_idx=peaks[0]
    first_spike = [cell_data.iloc[first_spike_idx],first_spike_idx]
    first_spike_data = first_spike[0]
    return first_spike

def extract_spike_frequency(trial_data,traial_no):
    sampling_rate=trial_data["sampling_rate(Hz)"].unique()[0]
    i_trace = trial_data["injected_current(pA)"].to_numpy()
    cell_trace = trial_data["cell_trace(mV)"].to_numpy()
    injected_current = np.max(i_trace)
    start_idx_injection = np.argmax(i_trace)
    end_idx_injection = start_idx_injection+int(0.25*sampling_rate) # 250 ms injection time for all protocols
    spike_trace=cell_trace[start_idx_injection:end_idx_injection]
    peaks, properties = scipy.signal.find_peaks(spike_trace, height=0)
    number_spikes = len(peaks)
    time_current_inj = (end_idx_injection-start_idx_injection)/sampling_rate
    spike_frequency = number_spikes/time_current_inj
    return spike_frequency, injected_current

def extract_spike_properties(cell_firing_data_all_cells,outdir):
    # Handle case where no firing data was found
    if cell_firing_data_all_cells.empty:
        print("Warning: No firing data to extract spike properties from. Creating empty properties DataFrame.")
        empty_props = pd.DataFrame(columns=['cell_ID', 'trial_no', 'spike_frequency', 'injected_current'])
        out_file = f"{outdir / 'all_cell_all_trial_firing_properties'}"
        write_pkl(empty_props, out_file)
        return empty_props
    
    cell_grp = cell_firing_data_all_cells.groupby(by="cell_ID")
    firing_properties=[]
    for cell, cell_data in cell_grp:
        trial_grp = cell_data.groupby(by="trial_no")
        for trial, trial_data in trial_grp:
            trial_df=pd.DataFrame()
            spike_frequency, injected_current = extract_spike_frequency(trial_data,trial)
            trial_df["cell_ID"]=[cell]
            trial_df["trial_no"]=trial
            trial_df["spike_frequency"]=spike_frequency
            trial_df["injected_current"]=injected_current
            firing_properties.append(trial_df)
    firing_properties = pd.concat(firing_properties).reset_index(drop=True)
    out_file = f"{outdir / 'all_cell_all_trial_firing_properties'}"
    write_pkl(firing_properties, out_file)
    return firing_properties

def convert_non_optical_data_to_pickle(folder_list_with_cell_data,cell_stats_h5_file,outdir):
    h_cells=list(cell_stats_h5_file.index)
    cell_firing_data_all_cells=[]
    for cell in tqdm(folder_list_with_cell_data):
        if cell.stem in h_cells:
            abf_list =list_files(cell)
            for abf in abf_list:
                protocol_name = protocol_file_name(abf)
                #print(protocol_name)
                if "cell_threshold" not in protocol_name:
                    continue
                else:
                    #print(protocol_name,abf)
                    firing_data_df = abf_to_df(abf)
                    firing_data_df.insert(loc=0, column='cell_ID', value=cell.stem)
                    cell_firing_data_all_cells.append(firing_data_df)
    
    # Check if any data was found before concatenation
    if not cell_firing_data_all_cells:
        print("Warning: No firing data found. Creating empty DataFrame.")
        # Create empty DataFrame with expected structure
        empty_df = pd.DataFrame(columns=['cell_ID', 'sampling_rate(Hz)', 'trial_no', 'cell_trace(mV)', 'injected_current(pA)', 'time(s)'])
        out_file = f"{outdir / 'all_cell_firing_traces'}"
        write_pkl(empty_df, out_file)
        return empty_df
    
    cell_firing_data_all_cells = pd.concat(cell_firing_data_all_cells,ignore_index=True)
    cell_firing_data_all_cells.reset_index(drop=True)
    out_file = f"{outdir / 'all_cell_firing_traces'}"
    write_pkl(cell_firing_data_all_cells, out_file)
    return cell_firing_data_all_cells


def main():
    # Argument parser.
    description = '''conversion script for abf files to hdf5.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells-path', '-f'
                        , required = False,default =None, type=str
                        , help = 'path of folder with folders as cell data '
                       )
    parser.add_argument('--cellstat-path', '-s'
                        , required = False,default =None, type=str
                        , help = 'path of folder with folders as cell data '
                       )

    args = parser.parse_args()
    
    # Dynamically find the repository root (where this script is located)
    script_dir = Path(__file__).parent  # conversion_scripts/
    repo_root = script_dir.parent        # main repository root
    
    # Set default paths if not provided
    if args.cells_path is None:
        p = repo_root / 'data' / 'cells_min_30mins_long'
    else:
    p = Path(args.cells_path)
    
    if args.cellstat_path is None:
        stats_path = repo_root / 'data' / 'hdf5_files' / 'abf_to_hdf5' / 'cell_stats.h5'
    else:
    stats_path = Path(args.cellstat_path)
    
    # Output to pickle_files directory with analysis tag
    outdir = repo_root / 'data' / 'pickle_files' / 'compile_firing_data'
    outdir.mkdir(exist_ok=True, parents=True)
    
    cells = list_folder(p)
    cell_stats_h5_file = pd.read_hdf(stats_path)
    cell_firing_data_all_cells = convert_non_optical_data_to_pickle(cells, cell_stats_h5_file, outdir)
    extract_spike_properties(cell_firing_data_all_cells, outdir)


if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main()
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
