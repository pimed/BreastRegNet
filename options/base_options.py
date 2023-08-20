"""
    Author: Negar Golestani
    Created: August 2023
"""


import argparse
import os
import torch
import copy

from .config import DATASET_ROOTDIR, CHECKPOINTS_ROOTDIR, RESULTS_ROOTDIR



####################################################################################################################################
class BaseOptions(object):
    def __init__(self):          
        self.initialized = False
    # ----------------------------------------------------------------------------------------------------
    def initialize(self):          
        self.parser = argparse.ArgumentParser(description='BrestRegNet PyTorch implementation')
        self.parser.add_argument('--dataset', type=str, default='sharpcut', help='Path to dataset dicetory; must have subfolders /real, /synthetic [sharpcut|sharpcut-mask|sharpcut-mask-reuse|sharpcut-blurmask|...]')                     
        self.parser.add_argument('--silent', action='store_true', help='if specified, stops displaying results and debugging information')                
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0,1,2. use -1 for CPU')
        self.parser.add_argument('--seed', type=int, default=42, help='Pseudo-RNG seed') 
        
        self.initialized = True         
    # ----------------------------------------------------------------------------------------------------
    def parse(self, save=False):           
        if not self.initialized: self.initialize()      # initialize options
        opt = self.parser.parse_args()                  # pars original options (saved)  
        opt_updated = self.set_extraOptions(opt)        # final options (not saved) 
        if save: self.save(opt, opt_updated.save_dir)   # save and display options

        self.opt = opt_updated
        return opt_updated   
    # ----------------------------------------------------------------------------------------------------
    def set_extraOptions(self, opt):
        opt_new = copy.deepcopy(opt)

        opt_new.type = self.__class__.__name__
        opt_new.dataset_rootdir = DATASET_ROOTDIR
        opt_new.checkpoints_rootdir = CHECKPOINTS_ROOTDIR
        opt_new.results_rootdir = RESULTS_ROOTDIR

        # update dataset info
        dataset_parts = opt_new.dataset.split('-')
        opt_new.dataset_name = dataset_parts[0]
        opt_new.reuse = 'reuse' in dataset_parts
        opt_new.blurmask = 'blurmask' in dataset_parts
        opt_new.mask = any(k in dataset_parts for k in ['mask', 'blurmask'])

        # Cuda
        if opt_new.gpu_id >= 0 and torch.cuda.is_available():
            opt_new.device = f"cuda:{opt_new.gpu_id}"
            torch.cuda.set_device(opt_new.gpu_id)
            torch.cuda.manual_seed(opt_new.seed)  # Seed
        else: opt_new.device = "cpu"

        self.opt = opt_new
        return opt_new             
    # ----------------------------------------------------------------------------------------------------
    def show(self, opt, display=True):  
        opt_str = ''
        opt_str += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            opt_str += '{:>25} : {:<30}{}\n'.format(str(k), str(v), comment)
        opt_str += '----------------- End -------------------'
        
        if display:
            print(opt_str)
            print("cuda:", torch.cuda.current_device())
        return opt_str
    # ----------------------------------------------------------------------------------------------------
    def save(self, opt, save_dir):
        opt_str = self.show(opt)
        if not os.path.exists(save_dir): os.makedirs(save_dir)            

        opt_path = os.path.join(save_dir, self.__class__.__name__ + '.txt' )
        with open( opt_path, 'wt') as file: file.write(opt_str)
####################################################################################################################################
