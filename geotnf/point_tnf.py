""" 
    Code adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

import torch
from .transformation import TpsGridGen



####################################################################################################################################
class PointTnf(object):
    """
        Class with functions for transforming a set of points with affine/tps transformations
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.tpsTnf = TpsGridGen(device=device)        
    # ----------------------------------------------------------------------------------------------------
    def tpsPointTnf(self, theta, points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        theta = theta.to(self.device)
        points = points.to(self.device)

        points = points.unsqueeze(3).transpose(1,3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        # undo reshaping
        warped_points=warped_points.transpose(3,1).squeeze(3)      
        return warped_points
    # ----------------------------------------------------------------------------------------------------
    def affPointTnf(self,theta,points):
        theta = theta.to(self.device)
        points = points.to(self.device)   

        theta_mat = theta.view(-1,2,3)
        # warped_points = torch.bmm(theta_mat[:,:,:2],points)
        # warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        warped_points = points + theta_mat[:,:,2].unsqueeze(2).expand_as(points)
        warped_points = torch.bmm(theta_mat[:,:,:2], warped_points)

        return warped_points 
####################################################################################################################################
def PointsToUnitCoords(P,im_size, device='cpu'):
    P = P.to(device)
    im_size = im_size.to(device)      
    h,w = im_size[:,0],im_size[:,1]

    NormAxis = lambda x,L: (x-1-(L-1)/2)*2/(L-1)
    P_norm = P.clone()
    # normalize X
    P_norm[:,0,:] = NormAxis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize Y
    P_norm[:,1,:] = NormAxis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm
####################################################################################################################################
def PointsToPixelCoords(P,im_size, device='cpu'):
    P = P.to(device)
    im_size = im_size.to(device)
    h,w = im_size[:,0],im_size[:,1]

    NormAxis = lambda x,L: x*(L-1)/2+1+(L-1)/2
    P_norm = P.clone()
    # normalize X
    P_norm[:,0,:] = NormAxis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize Y
    P_norm[:,1,:] = NormAxis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm
####################################################################################################################################
