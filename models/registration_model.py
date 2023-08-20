"""
    Author: Negar Golestani
    Created: August 2023
"""

import torch
import numpy as np
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from .base_model import baseModel
from .networks import*

pathLabels_types = ['DCIS', 'Invasive', 'Tumor Bed']


####################################################################################################################################
class regModel(baseModel): 
    # ----------------------------------------------------------------------------------------------------
    def set_network(self, network='regCorrNet-vgg16-layer4-1', device='cpu',  **kwargs):
        self.device = device
        
        network_parts = network.split('-')
        self.network_name = network_parts[0]
        network_info = '-'.join( network_parts[1:] ) if len(network_parts)>1 else None

        self.net = eval(self.network_name)(network_info=network_info, device=device, **kwargs)        
        self.discriminator = Discriminator(device=device)    
    # ----------------------------------------------------------------------------------------------------
    def set_optimizer(self, lr=0.0001, step_size=1, gamma=0.95, **kwargs):
        super().set_optimizer(lr, step_size, gamma, **kwargs)

        if 'adv' in self.LOSS.metrics:
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)
            self.scheduler_D = StepLR(self.optimizer_D, step_size=step_size, gamma=gamma)          
    # ----------------------------------------------------------------------------------------------------
    def set_inputs(self, input_data):
        '''
            INPUT PARAMS (NOTE [n=0,1,...])
                [REQUIRED] target_image, target_mask, source_image_n, source_mask_GT_n            
                [OPTIONAL] theta_GT_n, target_landmarks, source_landmarks_n, source_dpi, target_dpi 
        '''

        # Get number of source images
        if 'num_sources' in input_data:
            self.Nsource = input_data['num_sources']
        else:
            self.Nsource = 0
            while True:
                self.Nsource += 1
                if f'source_image_{self.Nsource}' not in input_data: break

        # data to batch (input data can be batch or single data)  
        self.batch = input_data.copy() 
        if input_data['target_image'].dim() == 3:
            for key, val in self.batch.items():
                try: self.batch[key] = val.unsqueeze(0)
                except: pass      
    # ----------------------------------------------------------------------------------------------------
    def forward(self, isTraining=False, **kwargs):
        '''
            ADDED PARAMS (NOTE [n=0,1,...])
                [REQUIRED] theta_n, warped_image_n, warped_mask_GT_n 
                [OPTIONAL] warped_dpi, warped_landmarks_n
        '''
        torch.cuda.empty_cache()
        
        if isTraining: self.net.train()
        else: self.net.eval()

        for n in range(self.Nsource): self.batch[f'theta_{n}'] = self.net( self.batch["target_image"], self.batch[f"source_image_{n}"], **kwargs) 
        # ------------------------------------------------

        self.warp()
    # ----------------------------------------------------------------------------------------------------
    def backward(self):
        if 'adv' in self.LOSS.metrics:
            # Train Discriminator
            self.optimizer_D.zero_grad()                            
            self.evalRes_tensor['loss_D'].backward(retain_graph=True)             

        # Train Generator (regNet)
        self.optimizer.zero_grad()                                          # set gradients to zero     
        self.evalRes_tensor['loss'].backward(retain_graph=True)             # calculate gradients
        self.optimizer.step()  

        if 'adv' in self.LOSS.metrics:
            self.optimizer_D.step()     # NOTE: both .step() have to be at the end 
    # ----------------------------------------------------------------------------------------------------
    def warp(self, input_data=None):
        if input_data is not None: self.set_inputs(input_data)        

        source_size = self.batch['source_image_0'].size()[2:]            

        # Delete previous warped data  
        for key in self.batch.keys():
            if 'warped' in key: self.batch.pop(key)  

        # dpi
        if 'source_dpi' in self.batch:
            self.batch['warped_dpi'] = np.array(self.batch['source_dpi']) 

        # images
        self.batch[f'warped_image'] = 0 
        for n in range(self.Nsource):
            self.batch[f'warped_image_{n}'] = self.net.geoTransformer(self.batch[f'source_image_{n}'], self.batch[ f'theta_{n}'], output_size=source_size)  
            self.batch[f'warped_image'] += self.batch[f'warped_image_{n}']


        # GT masks
        self.batch[f'warped_mask_GT'] = 0
        for n in range(self.Nsource):
            if f'source_mask_GT_{n}' in self.batch:
                self.batch[f'warped_mask_GT_{n}'] = self.net.geoTransformer( self.batch[f'source_mask_GT_{n}'], self.batch[ f'theta_{n}'], output_size=source_size, asBinary=True)  
                self.batch[f'warped_mask_GT'] += self.batch[f'warped_mask_GT_{n}']
        self.batch[f'warped_mask_GT'][self.batch[f'warped_mask_GT'] > 255.0] = 255.0


        # pathLabels
        for n in range(self.Nsource):
            for path_type in pathLabels_types:
                if f'source_pathLabel-{path_type}_{n}' in self.batch:
                    self.batch[f'warped_pathLabel-{path_type}_{n}'] = self.net.geoTransformer(self.batch[f'source_pathLabel-{path_type}_{n}'], self.batch[ f'theta_{n}'], output_size=source_size, asBinary=True)  
                    
                    if f'warped_pathLabel-{path_type}' in self.batch: self.batch[f'warped_pathLabel-{path_type}'] += self.batch[f'warped_pathLabel-{path_type}_{n}'] 
                    else: self.batch[f'warped_pathLabel-{path_type}'] = self.batch[f'warped_pathLabel-{path_type}_{n}']


        # landmarks (NOTE: only for test data with self.batch=1)
        warped_landmarks_list = list()
        for n in range(self.Nsource):
            if f'source_landmarks_{n}' in self.batch:
                self.batch[f'warped_landmarks_{n}'] = self.net.lmkTransformer(self.batch[f'source_landmarks_{n}'].squeeze(0), self.batch[ f'theta_{n}'].squeeze(0), source_size)
                warped_landmarks_list.append( self.batch[f'warped_landmarks_{n}'])
        if len(warped_landmarks_list)>0: self.batch['warped_landmarks'] = torch.cat(warped_landmarks_list)       

    # ----------------------------------------------------------------------------------------------------
    def get_evalRes(self, criterions):
        evalRes_tensor = dict()

        # resized batch (make source/earped image pixel_size same as target) > since target and source have same physical size > warped_dpi = target_dpi
        batch = dict()
        transform = T.Resize(size = self.batch['target_image'].size()[-2:])
        batch_size = self.batch['target_image'].size(0)

        for key, val in self.batch.items():
            if any([k in key for k in ['warped_image', 'warped_mask_GT']]): val = transform(val)
            batch[key] = val


        # Theta
        # if 'tnfGridLoss' in criterions: evalRes_tensor['tnfGridLoss'] = torch.sum(torch.stack( [self.tnfGridLoss(batch[f'theta_{n}'], batch[f'theta_GT_{n}']) for n in range(self.Nsource)]  ))

        # Images
        if 'mse' in criterions: evalRes_tensor['mse'] = self.MSE( batch['target_image'], batch['warped_image'] )            

        if 'ssim' in criterions: evalRes_tensor['ssim'] = self.SSIM( batch['target_image'], batch['warped_image'] )                                       
        if 'ssimLoss' in criterions: evalRes_tensor['ssimLoss'] = (1 - self.SSIM( batch['target_image'], batch['warped_image']) ) /2                                       
        
        if 'ncc' in criterions: evalRes_tensor['ncc'] = self.NCC( batch['target_image'], batch['warped_image'] )                                       
        if 'nccLoss' in criterions: evalRes_tensor['nccLoss'] = (1-self.NCC( batch['target_image'], batch['warped_image']) )/2                                       
        
        if 'mi' in criterions: evalRes_tensor['mi'] = self.MI( batch['target_image'], batch['warped_image'] )                                       
        if 'miLoss' in criterions: evalRes_tensor['miLoss'] = 1-self.MI( batch['target_image'], batch['warped_image'])                                       
        
        if 'hd' in criterions: evalRes_tensor['hd'] = self.HD( batch['target_image'], batch['warped_image'], batch['target_dpi'])                                       
        if 'hd95' in criterions: evalRes_tensor['hd95'] = self.HD95( batch['target_image'], batch['warped_image'], batch['target_dpi'])          


        # Masks
        if 'dice' in criterions: evalRes_tensor['dice'] = self.DICE( batch['target_mask_GT'], batch['warped_mask_GT'])    
        if 'diceLoss' in criterions: evalRes_tensor['diceLoss'] = 1-self.DICE( batch['target_mask_GT'], batch['warped_mask_GT'])   
        if 'diceSrc' in criterions: evalRes_tensor['diceSrc'] = self.DICE( batch['warped_mask_GT_0'], batch['warped_mask_GT_1'])   

        if 'sdm' in criterions: 
            warped_mask_GT_list = [batch[f'warped_mask_GT_{n}'] for n in range(self.batch['num_sources'])]            
            evalRes_tensor['sdm'] = self.SDM(batch['warped_dpi'], *warped_mask_GT_list)

        # Test Metrics  (NOTE: Only used for test dataset, but in form of batch with batch_size=1)
        # Landmarks            
        if 'mle' in criterions: 
            if all([k in batch for k in ['target_landmarks','warped_landmarks','target_dpi','warped_dpi']]):
                mle = self.MLE(batch['target_landmarks'].unsqueeze(0), batch['warped_landmarks'].unsqueeze(0), batch['target_dpi'], batch['warped_dpi'])                
                evalRes_tensor['mle'] = mle.nanmean()
                evalRes_tensor['mle_list'] = mle

             
        return evalRes_tensor                
####################################################################################################################################

