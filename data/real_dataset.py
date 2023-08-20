"""
Author: Negar Golestani
Created: August 2023
"""

import os
import pandas as pd
import torch
from pathlib import Path
from torch.autograd import Variable
import numpy as np
from PIL import Image

from .base_dataset import baseDataset


####################################################################################################################################
class realDataset(baseDataset):  
    landmarks_fn ='landmarks'
    pathLabels_fn ='labels'

    landmarks_format = '.csv'
    pathLabels_format = '.png'
    # ----------------------------------------------------------------------------------------------------
    def __getitem__(self, idx): 
        try:  
            # NOTE source_images must have the same size and dpi
            sample = dict()
            
            target_filename = self.filenames[idx]   
            sample['label'] = target_filename     

            sample['target_image'], sample['target_mask_GT'], sample['target_dpi'] = self.get_image(self.target_dir, target_filename)
            target_landmarks = self.get_landmarks(self.target_dir, target_filename)                    
            if len(target_landmarks): sample['target_landmarks'] = target_landmarks

            source_filename_list = self.filenames_df.loc[self.filenames_df[self.target_label]==target_filename][self.source_label].values
            sample['num_sources'] = len(source_filename_list)

            for n, source_filename_n in enumerate(source_filename_list):    
                sample[ f'source_image_{n}'], sample[ f'source_mask_GT_{n}'], sample[ f'source_dpi'] = self.get_image(self.source_dir, source_filename_n)
                source_landmarks_n = self.get_landmarks(self.source_dir, source_filename_n)
                if len(source_landmarks_n): sample[ f'source_landmarks_{n}'] = source_landmarks_n
            
                pathLabels = self.get_pathLabels(self.source_dir, source_filename_n)
                if len(pathLabels) > 0 : 
                    for name, mask in pathLabels.items(): sample[ f'source_pathLabel-{name}_{n}'] = mask

            return sample
    
        except: return None
    # ----------------------------------------------------------------------------------------------------
    def get_landmarks(self, load_dir, filename, as_npy=False):
        try:
            landmarks_path = str(Path(load_dir, self.landmarks_fn, filename).with_suffix(self.landmarks_format)) 
            landmarks_npy = pd.read_csv(landmarks_path)[['x','y']].values.astype('int')
        except:
            landmarks_npy = [] 

        if self.as_npy or as_npy: return landmarks_npy
        return Variable(torch.Tensor(landmarks_npy), requires_grad=False)   
    # ----------------------------------------------------------------------------------------------------
    def get_pathLabels(self, load_dir, filename, as_npy=False):        
        label_dir = os.path.join(load_dir, self.pathLabels_fn, filename)
        pathLabels = dict()
        if os.path.exists(label_dir):
            for fn in os.listdir(label_dir):
                fn = fn.split('.')[0]
                label_path = str(Path(label_dir, fn).with_suffix(self.pathLabels_format)) 
                label_PIL = Image.open(label_path).convert('1').convert('L')
                label_npy = np.repeat( np.expand_dims(np.array(label_PIL).astype(np.uint8), axis=2), 3, axis=2)

                if self.as_npy or as_npy: pathLabels[fn] = label_npy
                else: pathLabels[fn] = Variable(torch.Tensor(label_npy.transpose((2,0,1))), requires_grad=False).to(self.device) 
        
        return pathLabels
    # ----------------------------------------------------------------------------------------------------
    def get_cases(self, uniquate=True):        
        cases = [int(fn.split('_')[0]) for fn in self.filenames]        
        if uniquate: return np.unique(cases)
        return cases
####################################################################################################################################

