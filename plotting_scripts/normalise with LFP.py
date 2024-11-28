def field_norm_values(cell_list,val_to_plot):
    #cell_list is a pandas Df
    cell_list = cell_list.copy()
    #print(f"cell list inside func : {cell_list}")
    cell_grp=cell_list.groupby(by="cell_ID")
    for c, cell in cell_grp:
        pat_grp = cell.groupby(by="frame_id")
        for p,pat in pat_grp:
            #print(f"c:{c}, p:{p}")
            pre_val= float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]=="pre")][val_to_plot])
            field_val_pre=np.abs(float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]=="pre")]["min_field"]))
            pre_val_f= pre_val/field_val_pre
            pp_grp = pat.groupby(by="pre_post_status")
            for pr, pp in pp_grp:
                norm_val=float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]==pr)][val_to_plot])
                field_val_pr=np.abs(float(cell[(cell["cell_ID"]==c)&(cell["frame_id"]==p)&(cell["pre_post_status"]==pr)]["min_field"]))
                norm_val_f = norm_val/field_val_pr
                norm_val = (norm_val_f/pre_val_f)*100
                #norm_val = (norm_val/pre_val)*100
                cell_list.loc[(cell_list["cell_ID"]==c)&(cell_list["frame_id"]==p)&(cell_list["pre_post_status"]==pr),val_to_plot]=norm_val

    return cell_list