"""
    Author: Negar Golestani
    Created: August 2023
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from .utils.loss_functions import hd, hd95
from geotnf.point_tnf import PointTnf

InchMicron_ratio = 25400.
MicronMM_ratio = 0.001
InchMM_ratio = InchMicron_ratio * MicronMM_ratio



####################################################################################################################################
class Metric(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
    # ----------------------------------------------------------------------------------------------------    
    def forward(self, *args, make_imageBatch=False):
        args_new = list()
        for a in args:
            a_new = a if torch.is_tensor(a) else torch.Tensor(a) 
            if make_imageBatch and a.ndim==3: a_new = a_new.unsqueeze(0)
            a_new = a_new.to(self.device)                                   # Device (cuda/cpu)
            args_new.append(a_new)
        return args_new
####################################################################################################################################
class WeightedLoss(Metric):
    def __init__(self, weightedMetrics_str):
        super().__init__()

        self.metrics = list()
        self.weights = list()

        for wm in weightedMetrics_str.split('+'):
            if '*' in wm: w, m = wm.split('*')
            else: w, m = 1, wm

            self.metrics.append(m)
            self.weights.append(float(w))
    # ----------------------------------------------------------------------------------------------------
    def forward(self, evals_dict):    
        return torch.sum(torch.stack( [w*evals_dict[m] for (w, m) in zip(self.weights, self.metrics)] ))
####################################################################################################################################
class TransformedGridLoss(Metric):
    def __init__(self, geometric_model='affine', grid_size=120, device='cpu'):
        super().__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False).to(device)
        self.pointTnf = PointTnf(device=device)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, theta, theta_GT):
        # theta, theta_GT are batches

        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model=='affine':
            P_prime = self.pointTnf.affPointTnf(theta,P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)        
        elif self.geometric_model=='tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)
        
        return loss                
####################################################################################################################################
class MSE(Metric):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.mseLoss = nn.MSELoss()
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B):
        A_, B_ =  super().forward(A, B)
        A_, B_ = A_.div(255.0), B_.div(255.0)
        return  self.mseLoss(A_, B_)
####################################################################################################################################
class MLE(Metric):
    # Landmark Distance Error (mm)
    def forward(self, target_landmarks, warped_landmarks, target_dpi, warped_dpi):
        if warped_dpi is None: warped_dpi = target_dpi
        # landmarks are expected in [B,N,2] (torch)

        trg_lmk, wrp_lmk, trg_dpi, wrp_dpi = super().forward(target_landmarks, warped_landmarks, target_dpi, warped_dpi)
        if wrp_dpi is None: wrp_dpi = trg_dpi
        
        pxl2mm_target = torch.Tensor([ [float(InchMM_ratio/trg_dpi[0]), 0], [0, float(InchMM_ratio/trg_dpi[1])] ]).to(self.device)
        pxl2mm_warped = torch.Tensor([ [float(InchMM_ratio/wrp_dpi[0]), 0], [0, float(InchMM_ratio/wrp_dpi[1])] ]).to(self.device)
        pointsA = torch.matmul(trg_lmk, pxl2mm_target)
        pointsB = torch.matmul(wrp_lmk, pxl2mm_warped)

        # lmkDist = (pointsA - pointsB).pow(2).sum(dim=2).sqrt().sum(dim=0)
        lmkDist = (pointsA - pointsB).pow(2).sum(dim=-1).sqrt()
        if len(np.shape(lmkDist)) > 1: lmkDist = lmkDist.sum(dim=0)
        
        return lmkDist                  
####################################################################################################################################
class SSIM(Metric):
    '''  Reference: https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e '''
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, *args, window_size=11, channel=3, sigma=1.5, **kwargs):
        super().__init__(*args, **kwargs)

        # Create Window
        # self.window_size = min(window_size, H, W) # window should be atleast 11x11 
        self.window_size = window_size
        self.channel = channel
    
        self.window = self.create_window(self.window_size, channel, sigma=sigma).to(self.device)
    # ----------------------------------------------------------------------------------------------------
    def create_window(self, window_size, channel, sigma=1.5):
        gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        window_1D = (gauss/gauss.sum()).unsqueeze(1)

        # Converting to 2D  
        window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return torch.Tensor(window_2D.expand(channel, 1, window_size, window_size).contiguous())
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B):
        A_, B_ = super().forward(A, B)

        # Constants for stability 
        C1 = (0.01) ** 2  
        C2 = (0.03) ** 2         
        # L = 255   # dynamic range of pixel values (Removed here)

        pad = self.window_size // 2

        # Luminosity: calculate mu (locally) for both images using a gaussian filter 
        mu1 = nn.functional.conv2d(A_, self.window, padding=pad, groups=self.channel)
        mu2 = nn.functional.conv2d(B_, self.window, padding=pad, groups=self.channel)        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2

        # Contrast: calculate sigma square parameter
        sigma1_sq = nn.functional.conv2d(A_ * A_, self.window, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(B_ * B_, self.window, padding=pad, groups=self.channel) - mu2_sq
        sigma12 =  nn.functional.conv2d(A_ * B_, self.window, padding=pad, groups=self.channel) - mu12

        # SSIM 
        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        return ssim_score.nanmean()
####################################################################################################################################
class MI(Metric):
    # reference: https://github.com/connorlee77/pytorch-mutual-information
    def __init__(self, sigma=0.4, num_bins=256, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.sigma = 2*sigma**2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=self.device).float(), requires_grad=False)
    # ----------------------------------------------------------------------------------------------------
    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values
    # ----------------------------------------------------------------------------------------------------
    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf
    # ----------------------------------------------------------------------------------------------------
    def getMutualInformation(self, input1, input2):
        # Torch tensors for images between (0, 255)
        B, C, H, W = input1.size()
        assert((input1.size() == input2.size()))

        # Average over channels if images are not gray
        x1 = input1.mean(axis=1).view(B, H*W, 1)
        x2 = input2.mean(axis=1).view(B, H*W, 1)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
        
        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information
    # ----------------------------------------------------------------------------------------------------
    def getMutualInformation_old(self, input1, input2):
        # Torch tensors for images between (0, 255)
        B, C, H, W = input1.size()
        assert((input1.size() == input2.size()))
        
        x1 = input1.view(B, H*W, C)
        x2 = input2.view(B, H*W, C)

        # Average over channels if images are not gray
        x1 = x1.mean(axis=2).view(B, H*W, 1)
        x2 = x2.mean(axis=2).view(B, H*W, 1)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
        
        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B):
        A_, B_ =  super().forward(A, B, make_imageBatch=True)    
        return self.getMutualInformation(A_, B_).mean()
####################################################################################################################################
class Dice(Metric):
    def forward(self, A, B, eps=1e-8):    
        A_, B_ = super().forward(A, B)
        A_, B_ = A_.div(255.0), B_.div(255.0)

        A_ = A_.contiguous().view(-1)
        B_ = B_.contiguous().view(-1)        
        intersection = (A_ * B_).sum()                            
        dice = (2.*intersection)/(A_.sum() + B_.sum() + eps)      

        return dice  
####################################################################################################################################
class HD(Metric):
    # Reference: https://github.com/mavillan/py-hausdorff
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B, DPI): 
        # A, B are masks in shape of (B,C,H,W)
        A_, B_ = super().forward(A, B, make_imageBatch=True)

        v = 0
        for (m1, m2, dpi) in zip(A_, B_, DPI):
            m1 = m1.data.cpu().numpy().sum(axis=0)
            m2 = m2.data.cpu().numpy().sum(axis=0)
            spacing = InchMM_ratio/np.array(dpi)             
            v += hd(m1, m2, spacing)        
            
        return Variable(torch.Tensor( [v/A_.size()[0]]) , requires_grad=False).sum()
####################################################################################################################################
class HD95(Metric):
    # Reference: https://github.com/mavillan/py-hausdorff
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B, DPI): 
        # A, B are masks in shape of (B,C,H,W)
        A_, B_ = super().forward(A, B, make_imageBatch=True)

        v = 0
        for (m1, m2, dpi) in zip(A_, B_, DPI):
            m1 = m1.data.cpu().numpy().sum(axis=0)
            m2 = m2.data.cpu().numpy().sum(axis=0)
            spacing = InchMM_ratio/np.array(dpi)             
            v += hd95(m1, m2, spacing)        
            
        return Variable(torch.Tensor( [v/A_.size()[0]]) , requires_grad=False).sum()
####################################################################################################################################    
class NCC(Metric):
    # Reference: https://github.com/yuta-hi/pytorch_similarity
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, eps=1e-8, reduction='mean', **kwargs):
        super(NCC, self).__init__(**kwargs)
        self._eps = eps
        self._reduction = reduction
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B):
        shape = A.shape

        b = shape[0]
        # reshape
        A = A.contiguous().view(b, -1)
        B = B.contiguous().view(b, -1)
        # mean
        A_mean = torch.mean(A, dim=1, keepdim=True)
        B_mean = torch.mean(B, dim=1, keepdim=True)
        # deviation
        A = A - A_mean
        B = B - B_mean
        dev_AB = torch.mul(A,B)
        dev_AA = torch.mul(A,A)
        dev_BB = torch.mul(B,B)
        dev_AA_sum = torch.sum(dev_AA, dim=1, keepdim=True)
        dev_BB_sum = torch.sum(dev_BB, dim=1, keepdim=True)
        ncc = torch.div(dev_AB + self._eps / dev_AB.shape[1], torch.sqrt( torch.mul(dev_AA_sum, dev_BB_sum)) + self._eps)
        # ncc_map = ncc.view(b, *shape[1:])

        # reduce
        if self._reduction == 'mean': ncc = torch.mean(torch.sum(ncc, dim=1))
        elif self._reduction == 'sum': ncc = torch.sum(ncc)
        else: raise KeyError('unsupported reduction type: %s' % self._reduction)

        return ncc
####################################################################################################################################    
class MMD(Metric):
    def __init__(self, device='cpu'):
        super().__init__(device)
    # ----------------------------------------------------------------------------------------------------
    def forward_(self, A_features, B_features):                 
        # Flatten the feature maps
        A = torch.flatten(A_features, start_dim=1)
        B = torch.flatten(B_features, start_dim=1)

        # Compute the MMD loss between the two sets of feature maps
        sigma = np.median(A)
        A_A = torch.matmul(A, A.t())
        B_B = torch.matmul(B, B.t())
        A_B = torch.matmul(A, B.t())
        return  torch.mean(torch.exp(-A_A / (2 * sigma ** 2))) \
                + torch.mean(torch.exp(-B_B / (2 * sigma ** 2))) \
                - 2 * torch.mean(torch.exp(-A_B / (2 * sigma ** 2)))

        # mmd_loss = compute_mmd_loss(A_features, B_features)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B, kernel='multiscale'):       
        """
            Reference: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook 
            Emprical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.
            Args:
                x: first sample, distribution P
                y: second sample, distribution Q
                kernel: kernel type such as "multiscale" or "rbf"
        """
        x = torch.flatten(A, start_dim=1)
        y = torch.flatten(B, start_dim=1)

        x = nn.functional.normalize(x, p=2, dim=1)  # feature-wise normalization 
        y = nn.functional.normalize(y, p=2, dim=1)  # feature-wise normalization 


        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx 
        dyy = ry.t() + ry - 2. * yy 
        dxy = rx.t() + ry - 2. * zz         
        XX, YY, XY = (torch.zeros(xx.shape).to(self.device), torch.zeros(xx.shape).to(self.device), torch.zeros(xx.shape).to(self.device))
        
        if kernel == "multiscale":            
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if kernel == "rbf":        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        return torch.mean(XX + YY - 2. * XY)
####################################################################################################################################


 