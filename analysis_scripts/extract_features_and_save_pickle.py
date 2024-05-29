__author__           = "Anzal KS"
__copyright__        = "Copyright 2024-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import warnings
import time
from pathlib import Path
import argparse


class Args: pass
args_ = Args()

warnings.simplefilter(action='ignore', category=FutureWarning)

def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def write_pkl(file_to_write,file_path):
    with open(f'{file_path}.pickle', 'wb') as handle:
        pickle.dump(file_to_write,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(f" wrote file to pickle")
    
def substract_baseline(trace,sampling_rate, bl_period_in_ms):
    bl_period = bl_period_in_ms/1000
    bl_duration = int(sampling_rate*bl_period)
    bl = np.mean(trace[:bl_duration])
    bl_trace = trace-bl
    return bl_trace

def extract_cell_features_all_trials(cells_df, outdir):
    cell_grp = cells_df.groupby(by="cell_ID")
    cell_list = []
    for c, cell in tqdm(cell_grp):
        cell_rmp = np.mean(cell["cell_trace(mV)"])
        frame_status_grp = cell.groupby(by="frame_status")
        for fs, f_status in frame_status_grp:
            sampling_rate = int(f_status["sampling_rate(Hz)"].iloc[0])
            fit_i = int(0.003*sampling_rate) #default timepoint to start measuring slope=3ms
            fit_f = int(0.01*sampling_rate) #default timepoint to end measuring slope =10ms
            x_fit_f = int(0.01*sampling_rate) #dynamic timepoint to end measuring slope incase of rise time is too high =10ms
            total_tp = len(f_status["pre_post_status"].unique())
            pp_status_grp = f_status.groupby(by="pre_post_status")
            for pps, pp_status in pp_status_grp:
                if "pre" in pps:
                    time_point = -1
                elif "post" in pps:
                    time_point = int(pps.split("_")[-1])+1
                else:
                    continue
                pat_grp = pp_status.groupby(by='frame_id')
                for pat, patg in pat_grp:
                    if ("no_frame" in pat)or("inR" in pat):
                        continue
                    else:
                        trial_grp = patg.groupby(by='trial_no')
                        for tr, trial in trial_grp:
                            pat_trace = np.array(trial['cell_trace(mV)'])
                            field_trace = np.array(trial['field_trace(mV)'])
                            ttl_trace = np.array(trial["ttl_trace(V)"])
                            pat_trace = substract_baseline(pat_trace,sampling_rate,5) #5ms baseline
                            pat_trace = pat_trace[:int(sampling_rate*0.5)] #500ms time window
                            field_trace = substract_baseline(field_trace,sampling_rate,1) #1ms baseline
                            field_trace = field_trace[:int(sampling_rate*0.5)] #500ms time window
                            ttl_trace = ttl_trace[:int(sampling_rate*0.5)] #500ms time window
                            onset_time_idx = np.argmax(pat_trace>0.2)
                            abs_trace = np.abs(pat_trace)
                            pos_trace = pat_trace.clip(min=0)
                            neg_trace = np.abs(pat_trace.clip(max=0))
                            abs_area= np.trapz(abs_trace, dx=sampling_rate)
                            pos_area = np.trapz(pos_trace, dx=sampling_rate)
                            neg_area = np.trapz(neg_trace, dx=sampling_rate)
                            max_trace = float(pat_trace[np.argmax(pat_trace)])
                            min_trace = float(pat_trace[np.argmax(pat_trace==np.min(pat_trace[0:int(sampling_rate*0.2)]))]) # Epk is within 50ms, Ipk within 200ms
                            max_field = float(field_trace[np.argmax(field_trace)])
                            min_field = (field_trace[np.argmin(field_trace)])
                            time_x = np.linspace(0,len(pat_trace), int(sampling_rate*0.5))/sampling_rate
                            onset_time = float(time_x[onset_time_idx])
                            max_trace_t = float(time_x[np.where(pat_trace ==max_trace)[0][0]])
                            min_trace_t = float(time_x[np.where(pat_trace ==min_trace)[0][0]])
                            max_field_t = float(time_x[np.where(field_trace ==max_field)[0][0]])
                            min_field_t = float(time_x[np.where(field_trace ==min_field)[0][0]])
                            try:
                                fit_i = np.where(pat_trace>0.2)[0][0]
                                fit_f=np.where(pat_trace[:x_fit_f]<=0.66*(np.max(pat_trace)))[0][-1]
                                if fit_f<=fit_i:
                                    fit_i = int(0.005*sampling_rate)
                                    fit_f =int(0.01*sampling_rate)
                            except:
                                fit_i = int(0.005*sampling_rate)
                                fit_f =int(0.01*sampling_rate)
                            slope,intercept = np.polyfit(time_x[fit_i:fit_f],pat_trace[fit_i:fit_f],1)
                            cell_list.append([c,fs,pps,pat,tr,min_trace,
                                              max_trace,abs_area,pos_area,
                                              neg_area,onset_time,max_field,
                                              min_field,slope,intercept,
                                              min_trace_t,max_trace_t,
                                              max_field_t, min_field_t,
                                              pat_trace,field_trace,
                                              ttl_trace, cell_rmp])
    clist_header=["cell_ID","frame_status","pre_post_status","frame_id",
                  "trial_no","min_trace","max_trace","abs_area","pos_area",
                  "neg_area","onset_time","max_field","min_field","slope",
                  "intercept","min_trace_t","max_trace_t","max_field_t",
                  "min_field_t","trace","field","ttl","mean_rmp"]
    pd_cell_list =pd.concat(pd.DataFrame([i],columns=clist_header) for i in tqdm(cell_list))
    pd_cell_list = pd_cell_list[pd_cell_list["pre_post_status"]!="post_5"]
    outpath = f"{outdir}/pd_all_cells_all_trials"
    
    return pd_cell_list


def extract_cell_features_mean(all_data_with_training_df, outdir):
    cell_grp = all_data_with_training_df.groupby(by="cell_ID")
    cell_type_list = []
    for c, cell in tqdm(cell_grp):
        #print(f" colums: {cell.columns}")
        cell_rmp = np.mean(cell["cell_trace(mV)"])
        frame_status_grp = cell.groupby(by="frame_status")
        for fs, f_status in frame_status_grp:
            sampling_rate = int(f_status["sampling_rate(Hz)"].iloc[0])
            fit_i = int(0.003*sampling_rate) #default timepoint to start measuring slope=3ms
            fit_f = int(0.01*sampling_rate) #default timepoint to end measuring slope =10ms
            x_fit_f = int(0.01*sampling_rate) #dynamic timepoint to end measuring slope incase of rise time is too high =10ms
            #print(f" frame type: {fs } ::: sampling_rate = {sampling_rate}")
            total_tp = len(f_status["pre_post_status"].unique())
            pp_status_grp = f_status.groupby(by="pre_post_status")
            for pps, pp_status in pp_status_grp:
                if "training" in pps:
                    continue
                elif "pre" in pps:
                    time_point = -1
                else:
                    time_point = int(pps.split("_")[-1])+1
                pat_grp = pp_status.groupby(by='frame_id')
                for pat, patg in pat_grp:
                    if ("no_frame" in pat)or("inR" in pat):
                        continue
                    else:
                        pat_no = int(pat.split("_")[-1])
                        trial_grp = patg.groupby(by='trial_no')
                        mean_trace = []
                        mean_field = []
                        mean_ttl = []
                        for tr, trial in trial_grp:
                            pat_trace = np.array(trial['cell_trace(mV)'])
                            field_trace = np.array(trial['field_trace(mV)'])
                            ttl_trace = np.array(trial["ttl_trace(V)"])
                            #print(f"{sampling_rate}: {sampling_rate}")
                            pat_trace = substract_baseline(pat_trace,sampling_rate,5) #5ms baseline
                            pat_trace = pat_trace[:int(sampling_rate*0.5)] #500ms time window
                            field_trace = substract_baseline(field_trace,sampling_rate,1) #1ms baseline
                            field_trace = field_trace[:int(sampling_rate*0.5)] #500ms time window
                            ttl_trace = ttl_trace[:int(sampling_rate*0.5)] #500ms time window
                            mean_trace.append(pat_trace)
                            mean_field.append(field_trace)
                            mean_ttl.append(ttl_trace)
                        mean_trace = np.mean(np.array(mean_trace),axis=0)
                        mean_field = np.mean(np.array(mean_field),axis=0)
                        mean_ttl = np.mean(np.array(mean_ttl),axis=0)
                        
                        onset_time_idx = np.argmax(mean_trace>0.2)
                        abs_trace = np.abs(mean_trace)
                        pos_trace = mean_trace.clip(min=0)
                        neg_trace = np.abs(mean_trace.clip(max=0))
                        abs_area= np.trapz(abs_trace, dx=sampling_rate)
                        pos_area = np.trapz(pos_trace, dx=sampling_rate)
                        neg_area = np.trapz(neg_trace, dx=sampling_rate)
                        #print(f"total area = {abs_area}, pos_area={pos_area}, neg_area ={neg_area}")
                        
                        max_trace = float(mean_trace[np.argmax(mean_trace[0:int(sampling_rate*0.05)])])
                        #min_trace = mean_trace[np.where(mean_trace==np.min(mean_trace[int(sampling_rate*0.02):int(sampling_rate*0.2)]))[0][0]] # Epk is within 50ms, Ipk within 200ms
                        min_trace = float(mean_trace[np.argmax(mean_trace==np.min(mean_trace[0:int(sampling_rate*0.2)]))]) # Epk is within 50ms, Ipk within 200ms
                        
                        max_field = float(mean_field[np.argmax(mean_field)])
                        min_field = (mean_field[np.argmin(mean_field)])
                        
                        time_x = np.linspace(0,len(mean_trace), int(sampling_rate*0.5))/sampling_rate
                        onset_time = float(time_x[onset_time_idx])
                        
                        max_trace_t = float(time_x[np.where(mean_trace ==max_trace)[0][0]])
                        min_trace_t = float(time_x[np.where(mean_trace ==min_trace)[0][0]])
                        max_field_t = float(time_x[np.where(mean_field ==max_field)[0][0]])
                        min_field_t = float(time_x[np.where(mean_field ==min_field)[0][0]])
                        
                        try:
                            fit_i = np.where(mean_trace>0.2)[0][0]
                            fit_f=np.where(mean_trace[:x_fit_f]<=0.66*(np.max(mean_trace)))[0][-1]
                            #fit_f = np.where(trace <=0.66*(np.max(trace)))[0][-1]
                            if fit_f<=fit_i:
                                fit_i = int(0.005*sampling_rate)
                                fit_f =int(0.01*sampling_rate)
                        except:
                            fit_i = int(0.005*sampling_rate)
                            fit_f =int(0.01*sampling_rate)
                        slope,intercept = np.polyfit(time_x[fit_i:fit_f],mean_trace[fit_i:fit_f],1)
                        cell_type_list.append([c,fs,pps,pat,min_trace,
                                               max_trace,abs_area,pos_area,
                                               neg_area,onset_time,
                                               max_field, min_field,slope,
                                               intercept,min_trace_t,
                                               max_trace_t,max_field_t, 
                                               min_field_t,mean_trace,
                                               mean_field,mean_ttl, 
                                               cell_rmp])
    ctype_header=["cell_ID","frame_status","pre_post_status","frame_id",
                  "min_trace","max_trace","abs_area","pos_area","neg_area",
                  "onset_time","max_field","min_field","slope","intercept",
                  "min_trace_t","max_trace_t","max_field_t","min_field_t",
                  "mean_trace","mean_field","mean_ttl","mean_rmp"]
    pd_all_cells_mean =pd.concat(pd.DataFrame([i],columns=ctype_header) for i in tqdm(cell_type_list))
    outpath = f"{outdir}/pd_all_cells_mean"
    write_pkl(pd_all_cells_mean,"pd_all_cells_mean")
    return pd_all_cells_mean

#all_cell_trace_alone = all_cell_trace_alone[all_cell_trace_alone["cell_ID"]!="2022_12_12_cell_5"]
def sag_n_inR(trial_df_slice,sampling_rate):
    inj_current =-20
    input_R_trace = trial_df_slice['cell_trace(mV)'].to_numpy()
    trace_bl = np.mean(input_R_trace[0:int(0.15*sampling_rate)])
    ir_bl= np.mean(input_R_trace[int(0.45*sampling_rate):int(0.5*sampling_rate)])
    min_ir_trace_idx = np.argmin(input_R_trace[:int(0.35*sampling_rate)])
    min_ir_trace = input_R_trace[min_ir_trace_idx]
    sag = ((min_ir_trace-ir_bl)/inj_current)*1000
    inR = ((ir_bl-trace_bl)/inj_current)*1000
    #print(f"trace_bl, ir_bl,min_ir_trace, sag, inR: {trace_bl, ir_bl,min_ir_trace, sag, inR}")
    return sag,inR

def extract_cell_inR_features(cells_df,outdir):
    cell_grp = cells_df.groupby(by="cell_ID")
    cell_list_inR = []
    for c, cell in tqdm(cell_grp):
        pre_post_status_grp_data= cell.groupby(by="pre_post_status")
        for pp_type, pre_post_type_data in pre_post_status_grp_data:
            frame_status_grp = pre_post_type_data.groupby(by="frame_status")
            for fr_status, fr_status_data in frame_status_grp:
                if "training" in fr_status:
                    continue
                else:
                    sampling_rate=fr_status_data["sampling_rate(Hz)"].iloc[0]
                    frame_id_grp = fr_status_data.groupby(by="frame_id")
                    for fr_id, fr_id_data in frame_id_grp:
                        if "inR" in fr_id:
                            #print(fr_id)
                            fr_id_data = fr_id_data.loc[fr_id_data["frame_id"]=="inR"]
                            trial_grp_data = fr_id_data.groupby(by="trial_no")
                            for trial_no, trial_data in trial_grp_data:
                                sag, inR = sag_n_inR(trial_data,sampling_rate)
                                trace=trial_data['cell_trace(mV)'].to_numpy()
                                cell_list_inR.append([c,pp_type,fr_status,
                                                      trial_no,sag,inR,trace])
                        else:
                            continue
    cell_ir_list_header = ["cell_ID","pre_post_status","frame_status",
                           "trial_number","sag","inR","trace"]
    pd_cell_list_inR =pd.concat(pd.DataFrame([i],columns=cell_ir_list_header) for i in tqdm(cell_list_inR))
    pd_cell_list_inR_all_trials = pd_cell_list_inR[pd_cell_list_inR["pre_post_status"]!="post_5"]
    outpath = f"{outdir}/all_cells_inR"
    write_pkl(pd_cell_list_inR_all_trials,outpath)
    return pd_cell_list_inR

def cell_group_classifier(pd_all_cells_mean,outdir):
    cell_types_category = pd_all_cells_mean.groupby(by ="cell_ID")
    #cell_types_category =pd_all_cellsl_cells.groupby(by ="cell_ID")
    pot_cells =[]
    dep_cells =[]
    #ssp_cells =[]
    #ssn_cells =[]
    for c, cell in cell_types_category:
        cell["max_trace %"] = cell["max_trace"]
        cell["min_trace %"] = cell["min_trace"]
        cell["min_field %"] = cell["min_field"]
        pre_res_cutoff = cell[(cell["frame_id"]=="pattern_0")&(cell["pre_post_status"]=="pre")]["max_trace"][0]
        print(f"cutoff amplitude(0.5mv):{pre_res_cutoff}")
        if ((pre_res_cutoff<0.5) and pre_res_cutoff<10):
            print("disqualified")
            continue
        elif len(cell["pre_post_status"].unique())<6:
            print("rec length disq")
            continue
        else:
            print("qualified")
            pat_grp = cell.groupby(by="frame_id")
            for pa, pat in pat_grp:
                pps = pat["pre_post_status"].unique()
                pre_max_resp = pat[pat["pre_post_status"]=="pre"]["max_trace %"]
                pre_min_resp = pat[pat["pre_post_status"]=="pre"]["min_trace %"]
                pre_minf_resp = pat[pat["pre_post_status"]=="pre"]["min_field %"]
                for p in pps:
                    #print(f"p:{p}")
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"max_trace %"] = (pat[pat["pre_post_status"]==f"{p}"]["max_trace %"]/pre_max_resp)*100
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_trace %"] = (pat[pat["pre_post_status"]==f"{p}"]["min_trace %"]/pre_min_resp)*100
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_field %"] = (pat[pat["pre_post_status"]==f"{p}"]["min_field"]/pre_minf_resp)*100
                    cell_replaced_max=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["max_trace %"]
                    cell_replaced_min=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["min_trace %"]
                    cell_replaced_minf=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["min_field %"]
                    #print(f"pat: {pa}, pepo:{p}, cell_replaced:{cell_replaced}")
            pre_cell = np.array(cell[(cell["pre_post_status"]=="pre")&(cell["frame_id"]=="pattern_0")]["min_trace %"])
            #print(pre_cell)
            post_cell = np.array(cell[(cell["pre_post_status"]=="post_3")&(cell["frame_id"]=="pattern_0")]["max_trace %"])
            #cells with LTP and above 1mV
            if ((post_cell>pre_cell) and (post_cell>10)):
            #if post_cell>pre_cell:
                pot_cells.append(cell)
            #cells with depression and above 1mV
            #elif ((post_cell<pre_cell) and (post_cell>20)and (post_cell<600)):
            elif post_cell<pre_cell:
                dep_cells.append(cell)
            else:
                continue
            #cells with potentiation and less than 1mV 
            #elif ((post_cell>pre_cell) and (post_cell<10)and (post_cell<600)):
            #    ssp_cells.append(cell)
            ## cells with depression and less than 1mV
            #elif ((post_cell<pre_cell) and (post_cell<10)and (post_cell<600)):
            #    ssn_cells.append(cell)
    ap_cells_df = pd.concat(pot_cells).reset_index(drop=True)
    an_cells_df = pd.concat(dep_cells).reset_index(drop=True)
    all_cells = pd.concat([ap_cells_df,an_cells_df]).reset_index(drop=True)
    #ssp_cells_df = pd.concat(ssp_cells)
    #ssn_cells_df = pd.concat(ssn_cells)
    all_cells_dic = {"ap_cells": ap_cells_df,"an_cells":an_cells_df}#,"ssp_cells":ssp_cells_df,"ssn_cells":ssn_cells_df}
    outpath = f"{outdir}/all_cells_classified_dict"
    write_pkl(all_cells_dic,outpath)
    return all_cells,ap_cells_df,an_cells_df,all_cells_dic

def cell_classifier_with_fnorm(pd_all_cells_mean,outdir):
    #normalise with respect to feild and then classify
    cell_types_category = pd_all_cells_mean.groupby(by ="cell_ID")
    #cell_types_category =pd_all_cellsl_cells.groupby(by ="cell_ID")
    pot_cells_fnorm =[]
    dep_cells_fnorm =[]
    #ssp_cells_fnorm =[]
    #ssn_cells_fnorm =[]
    for c, cell in cell_types_category:
        cell["max_trace %"] = cell["max_trace"]
        cell["min_trace %"] = cell["min_trace"]
        cell["min_field %"] = cell["min_field"]
        cell["field_norm %"] = cell["max_trace"]
        pre_res_cutoff = cell[(cell["frame_id"]=="pattern_0")&(cell["pre_post_status"]=="pre")]["max_trace"][0]
        print(f"cutoff amplitude(0.5mv):{pre_res_cutoff}")
        if ((pre_res_cutoff<0.5) and pre_res_cutoff<10):
            print("disqualified")
            continue
        elif len(cell["pre_post_status"].unique())<6:
            print("rec length disq")
            continue
        else:
            print("qualified")
            pat_grp = cell.groupby(by="frame_id")
            for pa, pat in pat_grp:
                pps = pat["pre_post_status"].unique()
                pre_max_resp = pat[pat["pre_post_status"]=="pre"]["max_trace %"]
                pre_min_resp = pat[pat["pre_post_status"]=="pre"]["min_trace %"]
                pre_minf_resp = pat[pat["pre_post_status"]=="pre"]["min_field %"]
                pre_minf_resp_fnorm = pat[pat["pre_post_status"]=="pre"]["max_trace"]/np.abs(pat[pat["pre_post_status"]=="pre"]["min_field"])
                for p in pps:
                    #print(f"p:{p}")
                    f_norm_max = pat[pat["pre_post_status"]==f"{p}"]["field_norm %"]/np.abs(pat[pat["pre_post_status"]==f"{p}"]["min_field"])
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"max_trace %"] = (pat[pat["pre_post_status"]==f"{p}"]["max_trace %"]/pre_max_resp)*100
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_trace %"] = (pat[pat["pre_post_status"]==f"{p}"]["min_trace %"]/pre_min_resp)*100
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"min_field %"] = (pat[pat["pre_post_status"]==f"{p}"]["min_field %"]/pre_minf_resp)*100
                    cell.loc[(cell["frame_id"]==f"{pa}")&(cell["pre_post_status"]==f"{p}"),"field_norm %"] = (f_norm_max/pre_minf_resp_fnorm)*100
                    cell_replaced_max=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["max_trace %"]
                    cell_replaced_min=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["min_trace %"]
                    cell_replaced_minf=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["min_field %"]
                    cell_replaced_minfnorm=cell[(cell["pre_post_status"]==f"{p}")&(cell["frame_id"]==f"{pa}")]["field_norm %"]
                    #print(f"pat: {pa}, pepo:{p}, cell_replaced:{cell_replaced}")
            pre_cell = np.array(cell[(cell["pre_post_status"]=="pre")&(cell["frame_id"]=="pattern_0")]["field_norm %"])
            
            print(f'pre_cell:{pre_cell}')
            
            post_cell = np.array(cell[(cell["pre_post_status"]=="post_3")&(cell["frame_id"]=="pattern_0")]["field_norm %"])
            
            print(f'post cell:{post_cell}')
            
            #cells with LTP and above 1mV
            if post_cell>pre_cell:
            #if post_cell>pre_cell:
                pot_cells_fnorm.append(cell)
            #cells with depression and above 1mV
            #elif ((post_cell<pre_cell) and (post_cell>20)and (post_cell<600)):
            elif post_cell<pre_cell:
                dep_cells_fnorm.append(cell)
            else:
                continue
            #cells with potentiation and less than 1mV 
            #elif ((post_cell>pre_cell) and (post_cell<10)and (post_cell<600)):
            #    ssp_cells_fnorm.append(cell)
            ## cells with depression and less than 1mV
            #elif ((post_cell<pre_cell) and (post_cell<10)and (post_cell<600)):
            #    ssn_cells_fnorm.append(cell)
    ap_cells_df_fnorm = pd.concat(pot_cells_fnorm).reset_index(drop=True)
    an_cells_df_fnorm = pd.concat(dep_cells_fnorm).reset_index(drop=True)
    all_cells_fnorm = pd.concat([ap_cells_df_fnorm,an_cells_df_fnorm]).reset_index(drop=True)
    #ssp_cells_df = pd.concat(ssp_cells)
    #ssn_cells_df = pd.concat(ssn_cells)
    all_cells_dic_fnorm = {"ap_cells": ap_cells_df_fnorm,"an_cells":an_cells_df_fnorm}#,"ssp_cells":ssp_cells_df,"ssn_cells":ssn_cells_df}
    outpath = f"{outdir}/all_cells_fnorm_classifeied_dict"
    write_pkl(all_cells_dic_fnorm,outpath)
    return all_cells_fnorm,ap_cells_df_fnorm,an_cells_df_fnorm,all_cells_dic_fnorm




def main():
    # Argument parser.
    description = '''Opens all cell pickle and save extracted features'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pikl-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path to the giant pickle file with all cells'
                       )
    parser.add_argument('--cellstat-path', '-s'
                        , required = False,default ='./', type=str
                        , help = 'path to cell stats data'
                       )
    parser.add_argument('--outdir-path','-o'
                        ,required = False, default ='./', type=str
                        ,help = 'where to save the generated pickle files'
                       )
    #    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    pklpath = Path(args.pikl_path)
    statpath = Path(args.cellstat_path)
    globoutdir = Path(args.outdir_path)
    globoutdir= globoutdir/'pickle_files_from_analysis'
    globoutdir.mkdir(exist_ok=True, parents=True)
    
    cell_stats =pd.read_hdf(str(statpath))
    all_data_with_training_df = read_pkl(str(pklpath))
    
    extract_cell_inR_features(all_data_with_training_df,globoutdir)
    extract_cell_features_all_trials(all_data_with_training_df,globoutdir)
    pd_all_cells_mean = extract_cell_features_mean(all_data_with_training_df,globoutdir)
    cell_group_classifier(pd_all_cells_mean,globoutdir)
    cell_classifier_with_fnorm(pd_all_cells_mean,globoutdir)

if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
