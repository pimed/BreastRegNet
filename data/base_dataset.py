"""
    Author: Negar Golestani
    Created: August 2023
"""

from torch.utils.data import Dataset
import os
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import copy
import random
import torchvision
from pathlib import Path
import pandas as pd

Image.MAX_IMAGE_PIXELS = 100000000

####################################################################################################################################
class baseDataset(Dataset):  
    source_fn = 'source'
    target_fn = 'target'
    mask_fn = 'mask'

    target_label = 'target'
    source_label = 'source'

    image_format = '.tiff'
    mask_format = '.png'    
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, data_dir, *args, device='cpu', as_npy=False, empty=False, reuse=False, mask=False, blur_mask=False, **kwargs):
        self.data_dir = data_dir
        self.device = device
        self.as_npy = as_npy
        self.mask = mask
        self.blur_filter = torchvision.transforms.GaussianBlur(kernel_size=(11, 11), sigma=3.0) if mask and blur_mask else None


        self.target_dir = os.path.join(self.data_dir, self.target_fn)
        self.source_dir = os.path.join(self.data_dir, self.source_fn)
        self.filenames_df = pd.read_csv(os.path.join(data_dir, 'filenames.csv'))
        self.filenames = [] if empty else self.filenames_df.drop_duplicates('target')['target'].values

        if reuse: 
            self.source_dir = self.target_dir
            self.filenames_df = pd.DataFrame(data=dict(target=self.filenames, source=self.filenames))
    # ----------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.filenames)                               
    # ----------------------------------------------------------------------------------------------------
    def get_image(self, load_dir, filename, as_npy=False):        
        # image
        image_path = str(Path(load_dir, filename).with_suffix(self.image_format)) 
        image_PIL = Image.open(image_path).convert('RGB')
        image_npy = np.array(image_PIL).astype(np.uint8)    

        # dpi
        dpi = np.array(image_PIL.info['dpi']).astype('float')       

        # mask
        mask_path = str(Path(load_dir, self.mask_fn, filename).with_suffix(self.mask_format)) 
        mask_PIL = Image.open(mask_path).convert('1').convert('L')
        if self.blur_filter is not None: mask_PIL = self.blur_filter(mask_PIL)
        mask_npy = np.repeat( np.expand_dims(np.array(mask_PIL).astype(np.uint8), axis=2), 3, axis=2)

        if self.mask: image_npy = (image_npy * np.divide(mask_npy, 255.).astype('int')).astype('uint8')

        if self.as_npy or as_npy: 
            return image_npy, mask_npy, dpi

        image = Variable(torch.Tensor(image_npy.transpose((2,0,1))), requires_grad=False).to(self.device)      
        mask = Variable(torch.Tensor(mask_npy.transpose((2,0,1))), requires_grad=False).to(self.device) 

        return image, mask, dpi                        
    # ----------------------------------------------------------------------------------------------------
    def filter(self, exclude=None, include=None, label=None):
        cases = self.get_cases(uniquate=False)
        clone = copy.deepcopy(self)

        if include is not None: clone.filenames = [fn for (fn,case) in zip(clone.filenames, cases) if case in include] 
        if exclude is not None: clone.filenames = [fn for (fn,case) in zip(clone.filenames, cases) if case not in exclude] 
        if label is not None: clone.filenames = [fn for fn in clone.filenames if label in fn] 

        return clone
    # ----------------------------------------------------------------------------------------------------
    def partition(self, partitioning_info):
        if partitioning_info.lower == 'none':
            return [self], []

        param, partitioning_type = partitioning_info.split('-')   # Examples: k-fold, 0.2-split

        if partitioning_type == 'split': return self.partition_split( val_ratio=float(param) )
        elif partitioning_type =='fold': return self.partition_kfold( k=int(param) )
        elif partitioning_type =='foldFixed': return self.partition_kfoldFixed( k=int(param) )
    # ----------------------------------------------------------------------------------------------------
    def partition_split(self, val_ratio=.3):
                
        cases = self.get_cases() 
        random.shuffle(cases)    

        N_val = int( len(cases) * float(val_ratio) )

        train_dataset_list = [ self.filter(cases[:N_val]) ]
        val_dataset_list = [ self.filter(cases[N_val:]) ]

        return train_dataset_list, val_dataset_list 
    # ----------------------------------------------------------------------------------------------------
    def partition_kfold(self, k=5):
        cases = self.get_cases() 
        random.shuffle(cases)    
        N = len(cases)
        N_fold = int(N/k)

        train_dataset_list, val_dataset_list = list(), list()
        for i in range(k):
            a = i*N_fold
            b = (i+1)*N_fold if i < k-1 else N
            val_cases = cases[a:b]
            train_cases = list(set(cases)-set(val_cases))

            train_dataset_list.append( self.filter(exclude=val_cases) )
            val_dataset_list.append( self.filter(exclude=train_cases) )

        return train_dataset_list, val_dataset_list     
    # ----------------------------------------------------------------------------------------------------
    def partition_kfoldFixed(self, k=5):
        if k == 5:
            tr_cases_list = [
                [1, 5, 6, 7, 12, 13, 14, 15, 17, 18, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 40, 41, 42, 44, 47, 48, 49, 50],
                [1, 5, 6, 9, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 25, 26, 28, 30, 31, 32, 36, 37, 39, 40, 42, 44, 45, 48, 49],
                [5, 7, 9, 11, 13, 14, 17, 18, 20, 22, 24, 26, 27, 28, 30, 32, 33, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50],
                [1, 6, 7, 9, 11, 12, 14, 15, 20, 22, 23, 24, 25, 27, 28, 30, 31, 32, 33, 35, 36, 37, 39, 41, 42, 44, 45, 47, 48, 50],
                [1, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 20, 22, 23, 25, 26, 27, 31, 33, 35, 37, 39, 40, 41, 45, 47, 49, 50] ]
        
        else:
            print( f'> Error k-foldFixed cases for k={k} is not pre-defined')

        train_dataset_list = [self.filter(include=tr_cases) for tr_cases in tr_cases_list]
        val_dataset_list = [self.filter(exclude=tr_cases) for tr_cases in tr_cases_list]
    
        return train_dataset_list, val_dataset_list 
####################################################################################################################################
