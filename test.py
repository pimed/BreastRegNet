"""
    Author: Negar Golestani
    Created: August 2023
"""

import os
import numpy as np
from PIL import Image

from options.test_options import TestOptions
from utils.result_logger import Logger
from models import get_models, run_models
from data import get_dataset
import sys



####################################################################################################################################
def save_theta(warped_data, save_dir):
    parts = warped_data['label'].split('_')
    case, labels = parts[0], parts[1:]

    for n, label in enumerate(labels):
        tnf_save_path = os.path.join(save_dir, f"{case}_{label}.txt")
        with open(tnf_save_path, "w") as f:
            for line in list(warped_data[f"theta_{n}"].cpu().detach().numpy()):  f.write( f"{line}\n" )
####################################################################################################################################
def save_image(warped_data, save_dir, model_name):
    warped_image_npy = warped_data['warped_image'].data.cpu().numpy().transpose().astype(np.uint8).transpose((1,0,2)) 
    image_save_path = os.path.join(save_dir, f"{warped_data['label']}_{model_name}.tiff")
    Image.fromarray(warped_image_npy).save(image_save_path, dpi=list(warped_data['warped_dpi']))      
####################################################################################################################################
def save_mask(warped_data, save_dir, save_format='.png'):
    warped_image_npy = warped_data['warped_mask_GT'].data.cpu().numpy().transpose().astype(np.uint8).transpose((1,0,2)) 
    image_save_path = os.path.join(save_dir, f"{warped_data['label']}{save_format}")
    Image.fromarray(warped_image_npy).save(image_save_path, dpi=list(warped_data['warped_dpi']))      
####################################################################################################################################
def save_labels(warped_data, save_dir, save_format='.png'):
    save_dir_ = os.path.join(save_dir, warped_data['label'])

    for path_type in ['DCIS', 'Invasive', 'Tumor Bed']:
        if f'warped_pathLabel-{path_type}' not in warped_data: continue
        if not os.path.exists(save_dir_): os.makedirs(save_dir_)
        
        warped_image_npy = warped_data[f'warped_pathLabel-{path_type}'].data.cpu().numpy().transpose().astype(np.uint8).transpose((1,0,2)) 
        image_save_path = os.path.join(save_dir_, f"{path_type}{save_format}")
        Image.fromarray(warped_image_npy).save(image_save_path, dpi=list(warped_data['warped_dpi']))      
####################################################################################################################################
def save_labels_old(warped_data, save_dir, save_format='.png'):
    for path_type in ['DCIS', 'Invasive', 'Tumor Bed']:
        if f'warped_pathLabel-{path_type}' not in warped_data: continue
        warped_image_npy = warped_data[f'warped_pathLabel-{path_type}'].data.cpu().numpy().transpose().astype(np.uint8).transpose((1,0,2)) 
        image_save_path = os.path.join(save_dir, f"{warped_data['label']}_{path_type}{save_format}")
        Image.fromarray(warped_image_npy).save(image_save_path, dpi=list(warped_data['warped_dpi']))   
####################################################################################################################################

#------------------------------------------------------------------------------------------
if __name__ == "__main__":
#------------------------------------------------------------------------------------------
    opt = TestOptions().parse(save=True)   
    criterions = ['mle', 'dice', 'hd', 'hd95', 'ssim', 'mse', 'sdm']
    result_logger = Logger(opt.save_dir, 'test_evalRes')

    tnf_save_dir = os.path.join(opt.save_dir, 'tnf')
    if not os.path.exists(tnf_save_dir): os.makedirs(tnf_save_dir)
    mask_save_dir = os.path.join(opt.save_dir, 'mask')
    if not os.path.exists(mask_save_dir): os.makedirs(mask_save_dir)    
    labels_save_dir = os.path.join(opt.save_dir, 'labels')
    if not os.path.exists(labels_save_dir): os.makedirs(labels_save_dir)


    print('Creating Dataset ...')    
    test_dataset = get_dataset(opt)

    print('Loading Model ...')
    models = get_models(opt, version=-1)  

    print('Start testing ...')
    
    for data in test_dataset:
        if data is None: continue
        if data['num_sources'] > 5 : continue

        try:
            warped_data, evalRes = run_models( models, data, silent=opt.silent, criterions=criterions)  
            result_logger.log(evalRes, index=data['label'])       
            
            # ------- Save warped data -------
            save_image(warped_data, opt.save_dir, opt.names)
            save_theta(warped_data, tnf_save_dir)
            save_mask(warped_data, mask_save_dir)
            save_labels(warped_data, labels_save_dir)

             

        except: print( f"Error processing {data['label']}")

