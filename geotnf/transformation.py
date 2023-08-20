""" 
    Code adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision 

from .grid_gen import AffineGridGen, TpsGridGen



####################################################################################################################################
class GeometricTnf(object):
    """
        Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
        ( can be used with no transformation to perform bilinear resizing )        
    """
    def __init__(self, geometric_model='affine', out_h=224, out_w=224, device='cpu', minVal=0, maxVal=255):
        self.minVal = minVal
        self.maxVal = maxVal       

        self.out_h = out_h
        self.out_w = out_w
        self.device = device
        self.geometric_model = geometric_model

        if geometric_model=='affine': 
            self.gridGen = AffineGridGen(out_h, out_w)
        else:
            self.grid_size = int(geometric_model.split('_')[1])
            self.gridGen = TpsGridGen(out_h, out_w, grid_size=self.grid_size, device=device)

        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32)).to(self.device)
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, image_batch, theta_batch=None, padding_factor=0.0, crop_factor=1.0, output_size=None, asBinary=False):
        # out_szie = (out_h, out_w)
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3)
            theta_batch = Variable(theta_batch,requires_grad=False)  


        # if self.geometric_model=='affine': theta_batch = theta_batch.reshape(-1,2,3)
        # elif self.geometric_model=='tps': theta_batch = theta_batch.unsqueeze(2).unsqueeze(3)
        theta_batch = theta_batch.to(self.device)
        image_batch = image_batch.to(self.device)

        if output_size is not None and (output_size[0] != self.out_h or output_size[1] != self.out_w ):
            if self.geometric_model=='affine': gridGen = AffineGridGen(output_size[0], output_size[1])
            else: gridGen = TpsGridGen(output_size[0], output_size[1], grid_size=self.grid_size, device=self.device)
        else: gridGen = self.gridGen

        sampling_grid = gridGen( theta_batch )

        # warped_image_batch = F.grid_sample(image_batch, sampling_grid,padding_mode='zeros',  align_corners=True)
        warped_image_batch = F.grid_sample(image_batch.float(), sampling_grid, padding_mode='border',  align_corners=True)


        if asBinary:
            threshold = (self.minVal + self.maxVal)/2
            warped_image_batch[warped_image_batch < threshold] = self.minVal
            warped_image_batch[warped_image_batch >= threshold] = self.maxVal
        else:
            warped_image_batch[warped_image_batch < self.minVal] = self.minVal
            warped_image_batch[warped_image_batch > self.maxVal] = self.maxVal

        return warped_image_batch       
####################################################################################################################################    
class SynthPairTnf(object):
    def __init__(self, geometric_model='affine', crop_factor=1.0, output_size=(240,240), padding_factor=0.0, device='cpu', **kwargs):
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.device = device
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size 
        self.geometric_model = geometric_model

        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, device=self.device, **kwargs)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w, device=self.device, **kwargs)
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, batch, asBinary=False):
        # batch : {'imageA', 'imageB', 'theta'}
        # imageA -> no-change -> source
        # imafeB -> warped by theta -> target
        # theta -> ground truth theta         

        imageA_batch = batch['imageA'].to(self.device)
        imageB_batch = batch['imageB'].to(self.device)
        theta_batch = batch['theta']
        if theta_batch is not None: theta_batch.to(self.device)

        # convert to variables
        imageA_batch = Variable(imageA_batch, requires_grad=False)
        imageB_batch = Variable(imageB_batch, requires_grad=False)
        theta_batch =  Variable(theta_batch, requires_grad=False)       

        cropped_image_batch = self.rescalingTnf(imageA_batch, None, self.padding_factor, self.crop_factor,  asBinary=asBinary)              # Cropped image      

        warped_image_batch = self.geometricTnf(imageB_batch, theta_batch, self.padding_factor, self.crop_factor, asBinary=asBinary)         # Warped image    
  
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch }             
    # ----------------------------------------------------------------------------------------------------
    def symmetricImagePad(self,image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1)).to(self.device)
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1)).to(self.device)
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1)).to(self.device)
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1)).to(self.device)

        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch, image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch
####################################################################################################################################
    
  