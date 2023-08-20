"""
    Author: Negar Golestani
    Created: August 2023
"""

import argparse
import os
import pandas as pd

from .test_options import TestOptions
from .train_options import TrainOptions

####################################################################################################################################
def load_options(load_dir, setExtra=True):
    opt = argparse.Namespace()
    with open(load_dir, 'r') as f: content = f.readlines()
    
    for line in content[1:-1]:  
        line = line.strip().split('\t[default:')[0]
        idx =  line.find(':')
        key, val = line[:idx], line[idx+1:]
        # key, val = line.split(':')
        key, val = key.strip(), val.strip()
        
        if val == 'True': val_ = True
        elif val == 'False': val_ = False
        elif val.isdigit(): val_ = int(val)
        else:
            try: val_ = float(val)
            except: val_ = val
        setattr(opt, key, val_)
    
    if setExtra: 
        obj_type = os.path.basename(load_dir).split('.')[0]
        optObj = eval(obj_type)()
        opt = optObj.set_extraOptions(opt)

    return opt
####################################################################################################################################
def get_options_df(checkpoints_rootdir, opt_filename='TrainOptions.txt'):
    options_df = pd.DataFrame()
    for folder_path in os.scandir(checkpoints_rootdir):  
        trainOpt_path = os.path.join(folder_path.path, opt_filename)

        if os.path.exists(trainOpt_path): 
            loaded_opt = load_options(trainOpt_path, setExtra=False)
            options_df = options_df.append(loaded_opt.__dict__, ignore_index=True)

    return options_df
####################################################################################################################################
def options_exists(opt, options_df=None, checkpoints_rootdir=None, skip_keys=[]):        
    if options_df is None: df = get_options_df(checkpoints_rootdir) 
    else: df = options_df.copy()

    for key,val in opt.__dict__.items():
        if key not in df.columns or key in skip_keys: continue

        df = df[df[key]==val]
        if len(df) == 0: return False
    
    return True
####################################################################################################################################


