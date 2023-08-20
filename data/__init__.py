"""
Author: Negar Golestani
Created: August 2023
"""


import os
from .synth_dataset import synthDataset
from .real_dataset import realDataset



####################################################################################################################################
def get_dataset(opt, exclude=None, include=None, **kwargs):
    opt_dict = vars(opt)
    dataset_dir = os.path.join(opt.dataset_rootdir, opt.dataset_name)   

    with open(os.path.join(dataset_dir, 'DatasetOptions.txt') ) as f: contents = f.readlines()    
    dataset_type = {k:v for (k,v) in [[p.strip() for p in line.strip().split(':')] for line in contents[1:-1]]}['dataset_type']

    if dataset_type == 'synthetic': 
        dataset = synthDataset(dataset_dir, **opt_dict, **kwargs)
    elif dataset_type == 'real': 
        dataset = realDataset(dataset_dir, **opt_dict, **kwargs)       

    return dataset.filter(exclude=exclude, include=include)    
####################################################################################################################################
