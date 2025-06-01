__author__           = "Anzal KS"
__copyright__        = "Copyright 2022-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import neo.io as nio
import numpy as np
import pandas as pd
from scipy import signal as spy
from scipy import stats as stats
from pprint import pprint
import multiprocessing
import time
import argparse
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing

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

"""
1D array and get locations with a rapid rise, N defines the rolling window
"""
def find_ttl_start(trace, N):
    data = np.array(trace)
    #pulses= np.where( (data[1:]-data[:-1]) > 1)[0]-1 #(-1 just to offset by abit)
    #print(f"pulses : {pulses}")
    #"""
    data -= data.min()
    data /= data.max()
    pulses = []
    for i, x in enumerate(data[::N]):
        if (i + 1) * N >= len(data):
            break
        y = data[(i+1)*N]
        if x < 0.2 and y > 0.75:
            pulses.append(i*N)
    if pulses == []:
        #print("empty pulses")
        pulses = np.linspace(100,20100,100).astype(int)
        #print(pulses)
    else:
        pulses =pulses
        #print(pulses)
        #print("pulses not empty")
    #"""
    return pulses

"""
Convert channel names to index as an intiger
"""
def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

"""
function to find the protocol name for any abf file
"""
def protocol_file_name(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
    print(f"protocol_name:{protocol_name}.....")
    return protocol_name
        
"""
Detects the file name with training data (LTP protocol) in it 
"""
def training_finder(f_name):
    protocol_name = protocol_file_name(f_name)
#    print(f'protocol name = {protocol_name}')
    if 'training' in protocol_name:
        f_name= f_name
    elif 'Training' in protocol_name:
        f_name = f_name
#        print(f'training {f_name}')
    else:
#        print('not training')
        f_name = None
#    print(f'out_ training prot = {f_name}')
    return f_name 

"""
Sort the list of suplied files into pre and post trainign files and return the list 
"""
def pre_post_sorted(f_list):
    found_train=False
    for f_name in f_list:
        training_f = training_finder(f_name)
#        print(f'parsed prot train = {training_f}')
        if ((training_f != None) and (found_train==False)):
            training_indx = f_list.index(training_f)
            # training indx for post will have first element as the training protocol trace
            pre = f_list[:training_indx]
            post = f_list[training_indx:]
#            pprint(f'training file - {training_f} , indx = {training_indx} '
#                f'pre file ={pre} '
#                f'post file = {post} '
#                )
            found_train = True
        elif ((training_f != None) and (found_train==True)):
            no_c_train = f_name
        else:
            pre_f_none, post_f_none, no_c_train = None, None, None
    return [pre, post, no_c_train, pre_f_none, post_f_none ]

"""
Tag protocols with training, patterns, rmp measure etc.. assign a title to the file
"""
def protocol_tag(file_name):
    protocol_name = protocol_file_name(file_name)
    #print(f"protocol_name:{protocol_name}")
    if '12_points' in protocol_name:
        #print('point_protocol')
        title = 'Points'
    elif '42_points' in protocol_name:
        #print('point_protocol')
        title = 'Points'
    elif 'Baseline_5_T_1_1_3_3' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'patternsx' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'patterns_x' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'Training' in protocol_name:
        #print('training')
        title = 'Training pattern'
    elif 'training' in protocol_name:
        #print('training')
        title = 'Training pattern'
    elif 'training_pattern':
        #print("training")
        title = 'Training pattern'
    elif 'RMP' in protocol_name:
        #print('rmp')
        title='rmp'
    elif 'Input_res' in protocol_name:
        #print ('InputR')
        title ='InputR'
    elif 'threshold' in protocol_name:
        #print('step_current')
        title = 'step_current'
    else:
        #print('non optical protocol')
        title = None
    return title

"""
Pair files pre and post with point, patterns, rmp etc..
"""
def file_pair_pre_pos(pre_list,post_list):
    point = []
    pattern = [] 
    training = []
    rmp = []
    InputR = []
    step_current = []
    for pre in pre_list:
        tag = protocol_tag(pre)
#        print(f' tag on the file ={tag}')
        if tag=='Points':
            point.append(pre)
        elif tag=='Patterns':
            pattern.append(pre)
        elif tag =='rmp':
            rmp.append(pre)
        elif tag=='InputR':
            InputR.append(pre)
        elif tag =='step_current':
            step_current.append(pre)
        else:
            tag = None
            continue
    for post in post_list:
        tag = protocol_tag(post)
        if tag=='Points':
            point.append(post)
        elif tag=='Patterns':
            pattern.append(post)
        elif tag=='rmp':
            rmp.append(post)
        elif tag=='InputR':
            InputR.append(post)
        elif tag=='step_current':
            step_current.append(post)
        else:
            tag = None
            continue
#    print(f'point files = {point} '
#           f'pattern files = {pattern}'
#          )
    return [point, pattern,rmp, InputR, step_current]

"""
pair pre and post, points and patterns for each cell.
"""
def file_pair(cell_path): 
    cell_id = str(cell_path.stem)
    abf_list = list_files(cell_path)
    sorted_f_list = pre_post_sorted(abf_list)
    pre_f_list = sorted_f_list[0]
    post_f_list = sorted_f_list[1][1:]
    training_f = sorted_f_list[1][0]
    no_c_train = sorted_f_list[2]
    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
    paired_points = paired_list[0]
    paired_patterns = paired_list[1]
    #print(f"training_f: {training_f}")
    return [paired_points,paired_patterns,training_f]

"""
pattern label functions
"""
# plug in iteration umber and it returns a pattern type
def pat_selector(i):
    if i==0:
        pattern='Trained pattern'
    elif i==1:
        pattern='Overlapping pattern'
    elif i==2:
        pattern='Non overlapping pattern'
    else:
        pattern ='_NA'
    return pattern

def point_selector(i):
    if i<=4:
        point='Trained point'
    elif i>4:
        point='Untrained point'
    return point

"""
Injected_currentfinder
"""
def current_injected(reader):
    protocol_raw = reader.read_raw_protocol()
    protocol_raw = protocol_raw[0]
    protocol_trace = []
    for n in protocol_raw:
        protocol_trace.append(n[0])
    i_min = np.abs(np.min(protocol_trace))
    i_max = np.abs(np.max(protocol_trace))
    i_av = np.around((i_max-i_min),2)
    return i_av
"""
finding iput resistance of the cell from pandas df time series
"""
def input_R(grouped_cell_trace,injection_current ):
    #print(f" grouped_cell_trace.info:\n {grouped_cell_trace.info()}")
    grp_df = grouped_cell_trace.groupby(by=['frame_status'])
    input_R_po = None
    input_R_pat =None
    for gi, grp in grp_df:
        #print(f"gi:{gi}")
        if 'pattern' in gi:
            tf_pat = grp['time(s)'].iloc[-1]
            sampling_rate_pat = grp['sampling_rate(Hz)'].iloc[0]
            bl_pat = np.median(grp['cell_trace(mV)'].iloc[int(sampling_rate_pat*(tf_pat -0.7)):-1])
            dip_pat = np.median(grp['cell_trace(mV)'].iloc[int(sampling_rate_pat*(tf_pat- 1)):int(sampling_rate_pat*(tf_pat-0.8))])
            input_R_pat = np.absolute((bl_pat-dip_pat)/injection_current)*1000
            print(f'pattern: {tf_pat,sampling_rate_pat,bl_pat, dip_pat, input_R_pat}')
        elif 'point' in gi:
            tf_po = grp['time(s)'].iloc[-1]
            sampling_rate_po = grp['sampling_rate(Hz)'].iloc[0]
            bl_po = np.median(grp['cell_trace(mV)'].iloc[int(sampling_rate_po*(tf_po -0.7)):-1])
            dip_po = np.median(grp['cell_trace(mV)'].iloc[int(sampling_rate_po*(tf_po- 2.15)):int(sampling_rate_po*(tf_po-1.95))])
            input_R_po = np.absolute((bl_po-dip_po)/injection_current)*1000
            print(f'point: {tf_po,sampling_rate_po,bl_po,dip_po,input_R_po}')
        elif 'training' in gi:
            input_R_pat, input_R_po = 0,0
            print("input R calculation failed: non-optical protocol")
        else:
            input_R_pat, input_R_po = 0,0
            print("input R calculation failed: non-optical protocol")
    if (input_R_pat!=None)and(input_R_po!=None):
        input_R_pat = input_R_pat
        input_R_po = input_R_po
    else:
        input_R_pat=0 
        input_R_po =0
    print(f"inr: {input_R_pat, input_R_po}  ...############")
    return input_R_pat, input_R_po


"""
function to use on a cell's pandas df to check the cell health status
"""
def single_cell_health(cell_df):
    cut_off_vm = 1.5 # in mV
    cutt_off_rmp_ratio = 95 # in %
    InputR_cutt_off = 15 # in %
    injected_current = -20 # all protocols for input resistance I have injected -20pA
    #cell_df = cell_df.iloc[cell_df["frame_status"]!="training"]
    #print(f'unq frms : {cell_df["frame_status"].unique()}')
    cell_df =cell_df.loc[cell_df["frame_status"]!="training"].reset_index(drop=True) 
    #print(f'post selection unq frms: {cell_df["frame_status"].unique()}')
    rmp_median= cell_df['cell_trace(mV)'].median()
    rmp_std = np.around(cell_df['cell_trace(mV)'].std(),1)
    df_grp = cell_df.groupby(by=["pre_post_status","trial_no"])
    rmp_ratio = np.around((len(cell_df[cell_df['cell_trace(mV)'].between((rmp_median-cut_off_vm),(rmp_median+cut_off_vm))])/len(cell_df['cell_trace(mV)'])*100),1)
    input_R_pat = []
    input_R_po = []
    print(f'cell_ID:{cell_df["cell_ID"].unique()}.............')
    for name, group in df_grp:
        input_R_ =input_R(group, injected_current)
        input_R_pat.append(input_R_[0])
        input_R_po.append(input_R_[1])
    input_R_cell = np.array(input_R_pat+input_R_po)
    input_R_pattern=np.absolute(np.mean(input_R_pat[0:3]))
    input_R_point=np.absolute(np.mean(input_R_po[0:3]))
    input_R_cell_mean = np.mean(input_R_cell)
    inR_change = np.around((np.absolute((input_R_point-input_R_pattern)/input_R_pattern)*100),2)
    inR_cut = 0.15*input_R_pattern
    #print(f'in r cut{inR_cut}, inr change ={inR_change}, rmp ratio ={rmp_ratio}, rmp cut off = {cutt_off_rmp_ratio}')
    if (inR_change<InputR_cutt_off) and (rmp_ratio>cutt_off_rmp_ratio):
        cell_status='valid'
    else:
        cell_status='not_valid'
    #print(f'cell status = {cell_status}')
    cell_stats = {'InputR_cell_mean':input_R_cell_mean,'inR_cut':inR_cut,
                  'rmp_ratio':rmp_ratio, 'rmp_cut_off':cutt_off_rmp_ratio,
                 'cell_status':cell_status, 'inR_chancge':inR_change,
                  'inR_cell':input_R_cell,'rmp_median':rmp_median
                 }
    #print(f"cell status :  {cell_status}, inR_cut: {inR_cut}, rmp_ratio: {rmp_ratio}")
    return cell_stats

"""
convert abf to pandas df
"""
def abf_to_df(file_name,channel_name,pre_post_status):
    #fig,axs = plt.subplots(3,1, sharex=True,figsize=(6,2))
    #axs=axs.flatten()
    #global df_from_abf
    #df_from_abf = pd.DataFrame()
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
    injected_current = current_injected(reader)
    
    TTL_sig_all = []
    for s, segment in enumerate(segments):
        ttl_ch_no = channel_name_to_index(reader,'FrameTTL')
        ttl_signal = segment.analogsignals[ttl_ch_no]
        ttl_unit = str(ttl_signal.units).split()[1]
        ttl_trace = np.array(ttl_signal)
        TTL_sig_all.append(ttl_trace) 
        t = np.linspace(0,float(tf-ti),len(ttl_trace))
    ttl_av = np.average(TTL_sig_all,axis=0 )
    
    ttl_xi= find_ttl_start(ttl_av, 3)
    
    #print (f' TTL len = {len(ttl_xi)}')
    if ((len(ttl_xi)<5) and (len(ttl_xi)>2)):
        frame_type='pattern'
    elif ((len(ttl_xi)>5) and (len(ttl_xi)<60)):
        frame_type='point'
    else:
        frame_type="training"
        #print(f"*********** frame_type: {frame_type}")
    t=[]
    for s, segment in enumerate(segments):
        df_segment = pd.DataFrame()
        cell_ch_no = channel_name_to_index(reader,channel_name)
        field_ch_no = channel_name_to_index(reader,'Field')
        pd_ch_no = channel_name_to_index(reader,'Photodiode')
        ttl_ch_no = channel_name_to_index(reader,'FrameTTL')

        cell_signal = segment.analogsignals[cell_ch_no]
        cell_Signal_unit = str(cell_signal.units).split()[-1]
        cell_trace = np.hstack(np.ravel(np.array(cell_signal)))
        
        field_signal = segment.analogsignals[field_ch_no]
        field_signal_unit=str(field_signal.units).split()[-1]
        field_trace = np.hstack(np.ravel(np.array(field_signal)))
        
        pd_signal = segment.analogsignals[pd_ch_no]
        pd_signal_unit=str(pd_signal.units).split()[-1]
        pd_trace = np.hstack(np.ravel(np.array(pd_signal)))

        ttl_signal_unit=str(ttl_signal.units).split()[-1]
        ttl_trace = np.hstack(np.ravel(np.array(ttl_signal)))

        t = np.linspace(0,float(tf-ti),len(cell_trace))
        trial_no = s
        trial_no = [trial_no]*len(t)
        frame_status = [frame_type]*len(t)
        df_segment['frame_status']=frame_status
        ttl_status = ['no_frame']*len(t)
        ttl_starts = find_ttl_start(ttl_trace, 5)
        df_segment['frame_id']=ttl_status
        frame_segment_length = int(ttl_starts[1]-ttl_starts[0])
        for ttli,ttl_no in enumerate(ttl_starts):
            frame_status = f'{frame_type}_{ttli}'
            print(f"frame_status: {frame_status}")
            frame_status = [frame_status]*frame_segment_length
            df_segment['frame_id'].iloc[ttl_no:ttl_no+frame_segment_length]=frame_status
        inr_status=['inR']*len(df_segment[f'frame_id'][ttl_no+frame_segment_length:])
        df_segment['frame_id'].iloc[ttl_no+frame_segment_length:]=inr_status
        df_segment['trial_no']=trial_no
        df_segment[f'cell_trace({cell_Signal_unit})']=cell_trace
        df_segment[f'field_trace({field_signal_unit})']=field_trace
        df_segment[f'pd_trace({pd_signal_unit})']=pd_trace
        df_segment[f'ttl_trace({ttl_signal_unit})']=ttl_trace
        df_segment['time(s)']=t
        #inr_length=df_segment[df_segment["frame_id"]=="inR"][f'cell_trace({cell_Signal_unit})']
        #plt.plot(inr_length.to_numpy())
        #plt.show(block=False)
        #plt.pause(0.5)
        #plt.close()
        #print(df_segment.sample())
        df_from_abf.append(df_segment)
    df_from_abf = pd.concat(df_from_abf,ignore_index=True)
    df_from_abf.insert(loc=0, column=f'sampling_rate({sampling_rate_unit})', value=sampling_rate)
    df_from_abf.insert(loc=0, column='pre_post_status', value=pre_post_status)    
    #df_from_abf.insert(loc=0, column='frame_type', value=frame_type)

    
    #axs[0].plot(df_from_abf[f'cell_trace({cell_Signal_unit})'], 
    #            label=(f'cell_trace({cell_Signal_unit})'))
    #axs[1].plot(df_from_abf[f'ttl_trace({ttl_signal_unit})'], 
    #            label=(f'ttl_trace({ttl_signal_unit})'))
    #vline=df_from_abf[df_from_abf[f'ttl_trace({ttl_signal_unit})']>0.5][f'ttl_trace({ttl_signal_unit})'].index.values[1]
    #axs[2].plot(df_from_abf[f'pd_trace({pd_signal_unit})'], 
    #            label=(f'pd_trace({pd_signal_unit})'))
    ##axs[0].set_xlim(vline-100,vline+100)
    ##axs[1].set_xlim(vline-100,vline+100)
    ##axs[2].set_xlim(vline-100,vline+100)
    #plt.show(block=False)
    #plt.pause(0.5)
    #plt.close()

    return df_from_abf

"""
raw data from multiple abfs to dict, combine all the sorted abfs for a point or pattern to single nested dictionary
"""
def combine_abfs_for_one_frame_type(points_or_pattern_file_set_abf,cell_ID,ch_id='cell',training_status=None):
    #global all_frames_df
    if ch_id=='cell':
        ch_name='IN0'
    elif ch_id=='field':
        ch_name='Field'
    if training_status==False:
        #print(f"training_status: {training_status}")
        #print(f'ch Id = {ch_name}')
        pre_f = points_or_pattern_file_set_abf[0]
        post_f = points_or_pattern_file_set_abf[1:]
        #all_frames_df =pd.DataFrame()
        all_frames_df =[]
        all_frames_df.append(abf_to_df(pre_f,ch_name,'pre'))
        #all_frames_df.append(abf_to_df(training_f,ch_name,'training'))
        for ix,i in enumerate(post_f):
            m= abf_to_df(i,ch_name,f'post_{ix}')
            all_frames_df.append(m) # cell_all_frames_data.append(n)
        all_frames_df = pd.concat(all_frames_df,ignore_index=True)
        all_frames_df.insert(loc=0, column='cell_ID', value=cell_ID)
    elif training_status==True:
        training_f = points_or_pattern_file_set_abf
        all_frames_df =abf_to_df(training_f,ch_name,'training')
        all_frames_df = pd.concat([all_frames_df],ignore_index=True)
        all_frames_df.insert(loc=0, column='cell_ID', value=cell_ID)
    else:
        all_frames_df =None
    return all_frames_df
"""
combine all abfs related to one cell and give out a single pandas df from that for all frames (both patterns and points)

"""
def combine_frames_for_as_cell(cell_path,outdir):
    cell_ID = str(cell_path).split('/')[-1]
    files_paired = file_pair(cell_path)
    points_file_list = files_paired[0]
    patterns_file_list = files_paired[1]
    training_file= files_paired[2]
    points_raw_df = combine_abfs_for_one_frame_type(points_file_list,
                                                    cell_ID,ch_id='cell',
                                                    training_status=False )
    patterns_raw_df = combine_abfs_for_one_frame_type(patterns_file_list,
                                                      cell_ID,ch_id='cell',
                                                      training_status=False)
    training_raw_df = combine_abfs_for_one_frame_type(training_file,
                                                      cell_ID,ch_id='cell',
                                                     training_status=True)
    all_frames_raw_df =pd.concat([points_raw_df,patterns_raw_df,training_raw_df],ignore_index=True)
    cell_health = single_cell_health(all_frames_raw_df)
    cell_status_summary=[cell_ID, cell_health]
    all_frames_raw_df.to_hdf(f'{outdir}/{cell_ID}_with_training_data.h5',key='all_frames_raw_df')
    #print(cell_status_summary)
    return cell_status_summary 

def main():
    # Argument parser.
    description = '''conversion script for abf files to hdf5.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells-path', '-f'
                        , required = False,default =None, type=str
                        , help = 'path of folder with folders as cell data '
                       )


#    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    
    # Dynamically find the repository root (where this script is located)
    script_dir = Path(__file__).parent  # conversion_scripts/
    repo_root = script_dir.parent        # main repository root
    
    # Set default input path if not provided
    if args.cells_path is None:
        input_path = repo_root / 'data' / 'abf_all_cells'
    else:
        input_path = Path(args.cells_path)
    
    # Create output directory structure relative to repo root
    analysis_dir = repo_root / 'data' / 'hdf5_files' / 'abf_to_hdf5'
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # Cell data goes in all_cells_hdf subfolder
    outdir = analysis_dir / 'all_cells_hdf'
    outdir.mkdir(exist_ok=True, parents=True)
    
    # Cell stats file goes alongside the all_cells_hdf folder (not inside it)
    cell_stats_path = analysis_dir / 'cell_stats.h5'
    
    # Verify input path exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    cells = list_folder(input_path)
    cells_dict = pd.DataFrame()
    total_cells = len(cells)
    cell_stat =[] 
    
    print(f"Repository root: {repo_root}")
    print(f"Processing {total_cells} cells from: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Cell stats will be saved to: {cell_stats_path}")
    
    with multiprocessing.Pool(processes=6) as pool:
        futures = []
        for cell_no, cell in enumerate(cells):
            #print(f'Creating future for Cell {cell}')
            cell_ID = str(cell).split('/')[-1]
            if cell_ID == None:
                cell_ID = "invalid cell"
            else:
                cell_ID = cell_ID
            f = pool.apply_async(combine_frames_for_as_cell, args=(cell,outdir))
            futures.append(f)

        print('> Converting future to result')
        for i, fut in enumerate(futures):
            print(f' {i=}')
            stat = fut.get()
            cell_stat.append(stat)
            print(f'>saved {stat[0]} data')
    cell_stat = pd.DataFrame(cell_stat, columns=['cell_ID','cell_stats'])
    cell_stat.set_index('cell_ID',inplace = True)
    print(cell_stat)
    
    # Save cell stats to the correct path
    cell_stat.to_hdf(str(cell_stats_path), key='cell_stat')
    print(f"Cell statistics saved to: {cell_stats_path}")

if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main()
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
