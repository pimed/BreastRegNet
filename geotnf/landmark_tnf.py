""" 
    Code adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

import torch
from torch.autograd import Variable
import numpy as np
import cv2

from geotnf.point_tnf import PointsToUnitCoords, PointsToPixelCoords, PointTnf
from geotnf.transformation import GeometricTnf


 ########################################################################################################################
class LandmarkTnf(object):
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, device='cpu'):
        self.geoTransformer = GeometricTnf(geometric_model, out_h, out_w, device)
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, landmarks, theta, input_size, output_size=None):
        # landmarks are expected in [N,2] (torch) or [N,2] (numpy), where first column is X and second column is Y

        if output_size is None: output_size = input_size
        warped_landmarks = list()    
        for (x,y) in landmarks.data.cpu().numpy().reshape(-1,2): 
            try:
                point_img = cv2.circle(np.zeros((input_size[0], input_size[1], 3)), (int(x), int(y)), 10, (255,255,255), -1)      
                point_img = Variable(torch.Tensor(point_img.transpose((2,0,1))), requires_grad=False).unsqueeze(0)
                point_img_tnf = self.geoTransformer(point_img, theta, output_size=output_size)
                point_img_tnf = point_img_tnf.squeeze(0).data.cpu().numpy().transpose().astype(np.uint8).transpose((1,0,2))  

                Y, X = np.nonzero(point_img_tnf[:,:,0]!=0)
                x_new, y_new = int(np.nanmean(X)), int(np.nanmean(Y))            
            except: x_new, y_new = np.nan, np.nan
            warped_landmarks.append([x_new, y_new])
        
        return torch.Tensor( np.array(warped_landmarks) )  
########################################################################################################################
class LandmarkTnf_origfuncs(object):
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, geometric_model='affine', device='cpu'):
        # self.geoTransformer = GeometricTnf(geometric_model, out_h, out_w, device)
        self.geometric_model = geometric_model
        self.device = device
        self.pointTnf = PointTnf(device=device)

# ----------------------------------------------------------------------------------------------------
    def __call__(self, landmarks, theta_batch, input_size, output_size):
    # def landmarkTnf_orig_fncs(landmarks, tnf='affine', theta=None, input_size=(1000,1000), output_size=None, device='cpu'):
        # NOTE: not accurate !!!
        
        # landmarks are expected in [N,2] (torch) or [N,2] (numpy), where first column is X and second column is Y
        # When theta is None only transform landmarks from given input size to output size
        
        inp_size = Variable(torch.Tensor(np.array(input_size))).unsqueeze(0).to(self.device)
        out_size = inp_size if output_size is None else Variable(torch.Tensor(np.array(output_size))).unsqueeze(0).to(self.device) 

        warped_landmarks_batch = landmarks.transpose(1,0).unsqueeze(0).to(self.device)
        warped_landmarks_batch = PointsToUnitCoords( warped_landmarks_batch, inp_size, device=self.device)

        warped_landmarks_batch[:,1,:] *= -1 # Because direction of image pixels (in y direction) is different than XY coordinate system
        
        theta_batch = theta_batch.unsqueeze(0).to(self.device)        
        if self.geometric_model == 'affine': 
            theta_batch = theta_batch.view(1,6) * torch.Tensor([1,1,-1,1,1,1])
            warped_landmarks_batch = self.pointTnf.affPointTnf(theta_batch, warped_landmarks_batch)    
        elif self.geometric_model == 'tps': 
            warped_landmarks_batch = self.pointTnf.tpsPointTnf(theta_batch, warped_landmarks_batch)
        
        warped_landmarks_batch[:,1,:] *= -1 # Because direction of image pixels (in y direction) is different than XY coordinate system
        warped_landmarks_batch = PointsToPixelCoords(warped_landmarks_batch, out_size, device=self.device)

        # return warped landmarks in [N,2], where first column is X and second column is Y
        return warped_landmarks_batch.squeeze(0).transpose(1,0)
########################################################################################################################

