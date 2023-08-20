"""
    Author: Negar Golestani
    Created: August 2023
"""


import os
import torch
from collections import defaultdict
import random

from options import load_options
from .registration_model import regModel
from .registration_dual_model import regDualModel


####################################################################################################################################
def get_models(opt, version=None):
    if opt.type == 'TrainOptions':
        return eval(opt.model)(opt, version)
    # -------------------------------------------------------
    elif opt.type == 'TestOptions':
        models = list()
        for name in opt.name_list:            
            # Load training options
            checkpoints_path = os.path.join(opt.checkpoints_rootdir, name)   
            trainOpt = load_options(os.path.join(checkpoints_path, 'TrainOptions.txt') )

            # Build Model
            if version == -1: subfolders = [ f.path for f in os.scandir(checkpoints_path) if f.is_dir()]
            else: subfolders = [ os.path.join(checkpoints_path, str(version))]

            model_versions = list()
            for subfolder in subfolders:
                v = int(os.path.basename(subfolder))
                model = eval(trainOpt.model)(trainOpt, v)
                model.load_states()       
                model_versions.append(model)
                
            models.append(model_versions)
            
        return models
####################################################################################################################################
def run_models(models, data_, silent=True, **kwargs):
    data = data_.copy()

    for i, model_versions in enumerate(models):
        if i==0: warped_data = {k:v for k,v in data.items()}
        else: warped_data = { key.replace('warped', 'source'): val for key, val in warped_data.items() if 'source' not in key }                
        warped_data['num_sources'] = data['num_sources']

        for j, model in enumerate(model_versions): 
            if len(model_versions)>1 and int(data['label'].split('_')[0]) in model.tr_cases: continue      # Select the model not trained this data 
            warped_data, _ = model(warped_data, verbose=False,  **kwargs)    
            break

    # ------- Total Evaluation/Warping -------
    warped_data, evalRes = model(warped_data, verbose=not silent, calculate_theta=False,  **kwargs) 
    return warped_data, evalRes
####################################################################################################################################
def run_models_average(models, data, silent=True):
    for i, model_versions in enumerate(models):
        if i==0: warped_data = {k:v for k,v in data.items()}
        else: warped_data = { key.replace('warped', 'source'): val for key, val in warped_data.items() if 'source' not in key }                
        
        # Average theta from all versions/folds of a model
        theta = defaultdict(list)
        for j, model in enumerate(model_versions):
            warped_data_v, _ = model(warped_data, verbose=False)    
            for n in range(model.Nsource): theta[f'theta_{n}'].append( warped_data_v[f'theta_{n}'])

        # Warp by average of all versions 
        for i in range(model.Nsource): warped_data[f'theta_{i}'] = torch.mean(torch.stack(theta[f'theta_{i}'], dim=0), axis=0)                                
        warped_data, _ = model(warped_data, verbose=False, calculate_theta=False)      

        # Total theta (multi-stage)
        for i in range(model.Nsource): 
            if f'theta_{i}' in data:
                data[f'theta_{i}'] = torch.matmul( 
                    torch.cat([data[f'theta_{i}'].reshape((2,3)).to('cpu'), torch.Tensor([[0,0,1]])], axis=0),
                    torch.cat([warped_data[f'theta_{i}'].reshape((2,3)).to('cpu'), torch.Tensor([[0,0,1]])], axis=0) ).reshape(9,)[:6]
            else:
                data[f'theta_{i}'] = warped_data[f'theta_{i}']


    # ------- Total Evaluation/Warping -------
    warped_data, evalRes = model(data, verbose=not silent, calculate_theta=False)     
    return warped_data, evalRes
####################################################################################################################################
def run_models_random(models, data_, silent=True, num_random=4,  **kwargs):
    data = data_.copy()
    for num_source in range(20):
        if f"source_image_{num_source}" not in data: break # get number of source images
        data[f'theta_{num_source}'] = list()

    # random run
    for k in range(num_random):
        data_k = random_translation(models, data_, **kwargs)
        warped_data, _ = run_models(models, data_k, silent=True, **kwargs)
        for n in range(num_source): 
            data[f'theta_{n}'].append( warped_data[f'theta_{n}'] )

    # average of all estimated theta
    for n in range(num_source): 
        data[f'theta_{n}'] = torch.mean(torch.stack(data[f'theta_{n}']), dim=0)

    # ------- Total Evaluation/Warping -------
    warped_data, evalRes = models[0][0](data, verbose=not silent, calculate_theta=False,  **kwargs) 

    return warped_data, evalRes
####################################################################################################################################
def random_translation(models, data, move_range=.1, **kwargs):
    warped_data = {k:v for k,v in data.items()}
    
    for num_sources in range(20): 
        if f'source_image_{num_sources}' not in warped_data: break
        Tx =  random.randint(-move_range*1000, move_range*1000) * .001                       
        Ty =  random.randint(-move_range*1000, move_range*1000) * .001  
        warped_data[f"theta_{num_sources}"] = torch.Tensor([1.,0.,Tx,0.,1.,Ty])

    warped_data, _ = models[0][0](warped_data, verbose=False, calculate_theta=False, **kwargs)
    warped_data = { key.replace('warped', 'source'): val for key, val in warped_data.items() if 'source' not in key }                

    return warped_data
####################################################################################################################################
