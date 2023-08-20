"""
Author: Negar Golestani
Created: August 2023
"""


from math import cos, sin, pi
import torch
import random
import numpy as np

from .base_dataset import baseDataset
from geotnf.transformation import SynthPairTnf

####################################################################################################################################
class synthDataset(baseDataset):
    def __init__( self, data_dir, *args, geometric_model='affine', scale_range=0, rot_range=0, move_range=0, shear_range=0, image_h=224, image_w=224, random_t_tps=.3, **kwargs):
        super().__init__(data_dir, *args, **kwargs)

        self.geometric_model = geometric_model
        self.rot_range = rot_range
        self.move_range = move_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.random_t_tps = random_t_tps
        self.synthPairTnf = SynthPairTnf(geometric_model=self.geometric_model, device=self.device, output_size=(image_h, image_w))

        if 'tps' in self.geometric_model:
            self.grid_size = int(self.geometric_model.split('_')[1])
    #  ---------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):   
        sample = dict( label=self.filenames[idx] )

        target_filename = self.filenames[idx]        
        image, mask, _ = self.get_image(self.target_dir, target_filename)

        source_filename_list = self.filenames_df.loc[self.filenames_df[self.target_label]==target_filename][self.source_label].values

        for n, source_filename_n in enumerate(source_filename_list):    
            image_segment, mask_segment, _  = self.get_image(self.source_dir, source_filename_n)

            # theta
            theta = self.get_theta()   
            sample[ f'theta_GT_{n}'] = self.reverse_theta(theta)

            # image
            complete_sample_n = self.synthPairTnf({'imageA':image.unsqueeze(0), 'imageB':image.unsqueeze(0), 'theta':theta.unsqueeze(0)} )        
            segment_sample_n = self.synthPairTnf({'imageA':image_segment.unsqueeze(0), 'imageB':image_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)} )
            sample['target_image'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_image_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_image_{n}'] = segment_sample_n['target_image'].squeeze(0)

            # mask
            complete_sample_n = self.synthPairTnf({'imageA':mask.unsqueeze(0), 'imageB':mask.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)           
            segment_sample_n = self.synthPairTnf({'imageA':mask_segment.unsqueeze(0), 'imageB':mask_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)
            sample['target_mask_GT'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_mask_GT_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_mask_GT_{n}'] = segment_sample_n['target_image'].squeeze(0)
        
        return sample     
    #  ---------------------------------------------------------------------------------------------------------------
    def __getitem__old(self, idx):   
        sample = dict( label=self.filenames[idx] )

        target_filename = self.filenames[idx]        
        image, mask, _ = self.get_image(self.target_dir, target_filename)

        source_filename_list = self.filenames_df.loc[self.filenames_df[self.target_label]==target_filename][self.source_label].values

        for n, source_filename_n in enumerate(source_filename_list):    
            image_segment, mask_segment, _  = self.get_image(self.source_dir, source_filename_n)

            # theta
            theta = self.get_theta()     
            sample[ f'theta_GT_{n}'] = self.reverse_theta(theta)

            # image
            complete_sample_n = self.synthPairTnf({'imageA':image.unsqueeze(0), 'imageB':image.unsqueeze(0), 'theta':theta.unsqueeze(0)} )           
            segment_sample_n = self.synthPairTnf({'imageA':image_segment.unsqueeze(0), 'imageB':image_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)} )
            sample['target_image'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_image_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_image_{n}'] = segment_sample_n['target_image'].squeeze(0)

            # mask
            complete_sample_n = self.synthPairTnf({'imageA':mask.unsqueeze(0), 'imageB':mask.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)           
            segment_sample_n = self.synthPairTnf({'imageA':mask_segment.unsqueeze(0), 'imageB':mask_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)
            sample['target_mask_GT'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_mask_GT_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_mask_GT_{n}'] = segment_sample_n['target_image'].squeeze(0)
            
        return sample       
    # ----------------------------------------------------------------------------------------------------------------
    def get_theta(self):
        if self.geometric_model == 'affine':

            if self.scale_range == 0: 
                Sx, Sy = 1., 1.
            else: 
                Sx = random.randint( (1-self.scale_range)*1000, (1+self.scale_range)*1000) * .001       # scaling coefficients within [0.9 : 1.1]
                Sy = random.randint( (1-self.scale_range)*1000, (1+self.scale_range)*1000) * .001       # scaling coefficients within [0.9 : 1.1]
            rot = random.randint(-self.rot_range, self.rot_range) * pi/180                              # rotation angle within [-20 : 20] degrees
            Tx =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                    # shifting coefficients within 20% of image size
            Ty =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                    # shifting coefficients within 20% of image size
            Shx =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                 # shearing coefficients within 5% of image size
            Shy =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                 # shearing coefficients within 5% of image size

            theta = [   Sx*(cos(rot)-Shy*sin(rot)),   Sy*(Shx*cos(rot)-sin(rot)),    Tx, 
                        Sx*(sin(rot)+Shy*cos(rot)),   Sy*(Shx*sin(rot)+cos(rot)),    Ty   ]                        
        
        
        else:
            x = np.array([2*i/(self.grid_size-1)-1 for i in range(self.grid_size)]).reshape(-1,1)
            grd = np.array([*np.repeat(x,self.grid_size), *np.reshape([x]*self.grid_size, (1,-1))[0]]).reshape(-1,)
            theta = grd + (2*np.random.rand(2*self.grid_size*self.grid_size)-1)*self.random_t_tps 

        return torch.Tensor(theta)                
    # ----------------------------------------------------------------------------------------------------------------
    def get_theta_old(self, Ntps=6):
        if self.geometric_model == 'affine':

            if self.scale_range == 0: scale = 1
            else: scale = random.randint( (1-self.scale_range)*1000, (1+self.scale_range)*1000) * .001      # scaling coefficients within [0.9 : 1.1]
            rot = random.randint(-self.rot_range, self.rot_range) * pi/180                                  # rotation angle within [-20 : 20] degrees
            Tx =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                        # shifting coefficients within 20% of image size
            Ty =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                        # shifting coefficients within 20% of image size
            Sx =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                      # shearing coefficients within 5% of image size
            Sy =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                      # shearing coefficients within 5% of image size

            # theta = [scale*cos(rot), -scale*sin(rot), Tx, scale*sin(rot), scale*cos(rot), Ty]
            # theta = [   scale*(cos(rot)-Sx*sin(rot)), scale*(Sy*cos(rot)-sin(rot)), scale*Tx, 
            #             scale*(sin(rot)+Sx*cos(rot)), scale*(Sy*sin(rot)+cos(rot)), scale*Ty ]
            theta = [   scale*(cos(rot)-Sx*sin(rot)),   scale*(Sy*cos(rot)-sin(rot)),    Tx, 
                        scale*(sin(rot)+Sx*cos(rot)),   scale*(Sy*sin(rot)+cos(rot)),    Ty   ]                        
        
        
        elif self.geometric_model == 'tps':
            x = np.array([2*i/(Ntps-1)-1 for i in range(Ntps)]).reshape(-1,1)
            grd = np.array([*np.repeat(x,Ntps), *np.reshape([x]*Ntps, (1,-1))[0]]).reshape(-1,)
            theta = grd + (2*np.random.rand(2*Ntps*Ntps)-1)*.05 

        return torch.Tensor(theta)    
    #  ---------------------------------------------------------------------------------------------------------------
    def reverse_theta(self, theta):
        if self.geometric_model == 'affine':
            return torch.cat( [theta.view(6), torch.tensor([0, 0, 1])]).view(3,3).inverse().reshape(9)[:6]
        
        else: 
            return torch.Tensor([])
    #  ---------------------------------------------------------------------------------------------------------------
    def get_cases(self, uniquate=True):
        cases = [int(fn.split('_')[1]) for fn in self.filenames]

        if uniquate: return np.unique(cases)
        return cases
####################################################################################################################################
