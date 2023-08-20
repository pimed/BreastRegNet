"""
    Author: Negar Golestani
    Created: August 2023
"""
  

import torch
import torch.nn as nn
from torch.autograd import Variable


from .utils import *
from geotnf.transformation import GeometricTnf
from geotnf.landmark_tnf import LandmarkTnf


####################################################################################################################################
#              Registration
####################################################################################################################################     
class regCorrNet(nn.Module):  
    def __init__(self, network_info, geometric_model='affine', image_h=224, image_w=224, device='cpu', theta_lr=.1, **kwargs):
        super().__init__()

        # initialization
        encoder_info, last_layer, requires_grad_code = network_info.split('-')
        encoder_parts = encoder_info.split('_')
        encoder_name = encoder_parts[0]
        attention = encoder_parts[1].upper() if len(encoder_parts)>1 else None

        # build
        input_size = (1,3,image_h,image_w)      
        self.FeatureExtraction = load_pretrained_model( encoder_name, last_layer, attention=attention, 
                                                        requires_grad_code=requires_grad_code, input_size=input_size).to(device)
        self.geoTransformer = GeometricTnf(geometric_model=geometric_model, out_h=image_h, out_w=image_w, device=device)
        self.lmkTransformer = LandmarkTnf(geometric_model=geometric_model, out_h=image_h, out_w=image_w, device=device)
        
        test_data = Variable(torch.zeros(input_size), requires_grad=False).float().to(device)
        feature_size = self.FeatureExtraction(test_data).size()
        self.FeatureRegression = CorrCNN(geometric_model, feature_size, device)              
        self.TnfStabilizer = TnfStabilizer(geometric_model, device, theta_lr=theta_lr)     
    # ----------------------------------------------------------------------------------------------------
    def forward(self, target_image, source_image):
        source_image_ = self.geoTransformer(source_image).div(255.0)         
        target_image_ = self.geoTransformer(target_image).div(255.0)         

        feature_A = self.FeatureExtraction(source_image_)
        feature_B = self.FeatureExtraction(target_image_)

        theta = self.FeatureRegression(feature_A, feature_B)
        theta = self.TnfStabilizer(theta)

        return theta
####################################################################################################################################
class regCorrDualNet(nn.Module):  
    def __init__(self, network_info, geometric_model='affine', image_h=224, image_w=224, device='cpu', theta_lr=.1, **kwargs):
        super().__init__()

        # initialization
        encoder_info, last_layer, requires_grad_code = network_info.split('-')
        encoder_parts = encoder_info.split('_')
        encoder_name = encoder_parts[0]
        attention = encoder_parts[1].upper() if len(encoder_parts)>1 else None

        # build
        input_size = (1,3,image_h,image_w)      
        self.source_FeatureExtraction = load_pretrained_model( encoder_name, last_layer, attention=attention, requires_grad_code=requires_grad_code, input_size=input_size).to(device)
        self.target_FeatureExtraction = load_pretrained_model( encoder_name, last_layer, attention=attention, requires_grad_code=requires_grad_code, input_size=input_size).to(device)
        self.geoTransformer = GeometricTnf(geometric_model=geometric_model, out_h=image_h, out_w=image_w, device=device)
        self.lmkTransformer = LandmarkTnf(geometric_model=geometric_model, out_h=image_h, out_w=image_w, device=device)
        
        test_data = Variable(torch.zeros(input_size), requires_grad=False).float().to(device)
        feature_size = self.source_FeatureExtraction(test_data).size()
        self.FeatureRegression = CorrCNN(geometric_model, feature_size, device)              
        self.TnfStabilizer = TnfStabilizer(geometric_model, device, theta_lr=theta_lr)     
    # ----------------------------------------------------------------------------------------------------
    def forward(self, target_image, source_image, input_type='multi'):
        source_image_ = self.geoTransformer(source_image).div(255.0)         
        target_image_ = self.geoTransformer(target_image).div(255.0)         

        if input_type == 'source':
            feature_A = self.source_FeatureExtraction(source_image_)
            feature_B = self.source_FeatureExtraction(target_image_)

        elif input_type == 'target':
            feature_A = self.target_FeatureExtraction(source_image_)
            feature_B = self.target_FeatureExtraction(target_image_)   

        else:        
            feature_A = self.source_FeatureExtraction(source_image_)
            feature_B = self.target_FeatureExtraction(target_image_)

        theta = self.FeatureRegression(feature_A, feature_B)
        theta = self.TnfStabilizer(theta)

        return theta       
####################################################################################################################################




