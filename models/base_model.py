"""
    Author: Negar Golestani
    Created: August 2023
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time

from .loss import*
from .networks import*


####################################################################################################################################
class baseModel(nn.Module):
    def __init__(self, opt, version=None):
        super().__init__()
        # Checkpoints directory
        self.save_dir = os.path.join(opt.checkpoints_rootdir, opt.name)   # save all the checkpoints to checkpoints_rootdir
        if version is not None: self.save_dir = os.path.join(self.save_dir, str(version))          
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir) 
       
        # Initialization
        opt_dict = vars(opt)
        self.set_criterions(**opt_dict)
        self.set_network(**opt_dict)
        self.set_optimizer(**opt_dict)

        # Parameters
        self.epoch = 0
        self.tr_cases = []   
        torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, test_data, verbose=True, criterions=['mle'], calculate_theta=True, **kwargs):
        start_time = time.time()
        self.set_inputs(test_data)                                              # set inputs

        if calculate_theta:
            self.forward(isTraining=False, **kwargs)                            # forward pass

        et = time.time() - start_time                                           # execution time
        evalRes = self.eval(criterions=criterions)                              # calculate eval results    
        evalRes['et'] = et


        # Display Results     
        if verbose: 
            label = test_data['label'] if 'label' in test_data else ''
            self.print_results(label, evalRes)

        # Output data from batch
        output_data = dict()
        for key, val in self.batch.items():
            try: output_data[key] = val.squeeze(0)
            except: output_data[key] = val

        # torch.cuda.empty_cache()
        return output_data, evalRes 
    # ====================================================================================================
    #                                       PLACEHOLDERS
    # ====================================================================================================
    def set_network(self, *args, **kwargs):
        ''' Placeholder '''
        pass
    # ----------------------------------------------------------------------------------------------------    
    def set_inputs(self, *args, **kwargs):
        ''' Placeholder '''
        pass
    # ----------------------------------------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        ''' Placeholder '''
        pass  
    # ----------------------------------------------------------------------------------------------------
    def get_evalRes(self, *args, **kwargs):
        ''' Placeholder '''
        pass
    # ====================================================================================================  
    # ====================================================================================================   
    def set_criterions(self, loss='mse', geometric_model='affine', grid_size=120, device='cpu', **kwargs):
        ''' Placeholder '''
        self.tnfGridLoss = TransformedGridLoss(geometric_model=geometric_model, grid_size=grid_size, device=device) 
        self.MSE = MSE(device=device)
        self.SSIM = SSIM(device=device)
        self.MI = MI(device=device)
        self.DICE = Dice(device=device) 
        self.MLE = MLE(device=device)
        self.HD = HD(device=device)
        self.HD95 = HD95(device=device)
        self.NCC = NCC(device=device)
        self.MMD = MMD(device=device)
        self.GANloss = nn.BCELoss()
        self.LOSS = WeightedLoss(loss)
    # ----------------------------------------------------------------------------------------------------
    def set_optimizer(self, lr=1e-4, step_size=1, gamma=.95, **kwargs):
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    # ----------------------------------------------------------------------------------------------------
    def backward(self):
        self.optimizer.zero_grad()                                          # set gradients to zero     
        self.evalRes_tensor['loss'].backward(retain_graph=True)             # calculate gradients
        self.optimizer.step()                                               # update weights                
    # ----------------------------------------------------------------------------------------------------
    def eval(self, criterions=[]): 
        self.evalRes_tensor = self.get_evalRes(criterions)

        try: self.evalRes_tensor['loss'] = self.LOSS( self.evalRes_tensor )  
        except: self.evalRes_tensor['loss'] = None

        # evalRes (numpy)
        evalRes_npy = dict()
        for key, val in  self.evalRes_tensor.items(): 
            try:  val = val.data.cpu().numpy()
            except: pass
            
            if val is not None: evalRes_npy[key] = val

        return evalRes_npy                                                                  
    # ----------------------------------------------------------------------------------------------------
    def run_singleEpoch(self, dataset, isTraining=False, num_workers=4, batch_size=1, shuffle=False, criterions=[]):   
        criterions_ = [*criterions, *self.LOSS.metrics]                             # make sure all criterions for loss are considered to be calculated
        evalRes_epoch = pd.DataFrame()                                              # epoch eval
        

        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers):
            self.set_inputs(batch)
            self.forward(isTraining=isTraining)                                     # forward pass
            evalRes_batch = self.eval(criterions=criterions_)                       # calculate eval results 
            if isTraining: self.backward()                                          # get gradients and update network weights 
            evalRes_epoch = evalRes_epoch.append(evalRes_batch, ignore_index=True)  # Batch evaluation

        evalRes_epoch = evalRes_epoch.mean().to_dict()                              # Results (epoch evaluation)
        # torch.cuda.empty_cache()
        return evalRes_epoch         
    # ----------------------------------------------------------------------------------------------------
    def train(self, train_dataset, verbose=True, **kwargs):
        evalRes_epoch = self.run_singleEpoch(train_dataset, isTraining=True, **kwargs)
        self.scheduler.step() # Update scheduler and epoch
        self.epoch += 1  
        self.tr_cases = list( set(self.tr_cases).union(set(train_dataset.get_cases())) )
        
        # Display Results 
        if verbose: 
            print( f'Epoch {self.epoch}' )     
            self.print_results('Train', evalRes_epoch)

        return evalRes_epoch
    # ----------------------------------------------------------------------------------------------------
    def validate(self, test_dataset, verbose=True, **kwargs):
        evalRes_epoch = self.run_singleEpoch( test_dataset, isTraining=False, **kwargs)  
        if verbose: self.print_results('Test', evalRes_epoch)   # Display Results     

        return evalRes_epoch
    # ----------------------------------------------------------------------------------------------------
    def save_states(self, filename='net'):       
        save_filename = f"{filename}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        state = {
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),         
            'epoch': self.epoch,
            'tr_cases': self.tr_cases   }        

        torch.save(state, save_path)       
    # ----------------------------------------------------------------------------------------------------
    def load_states(self, filename='net'):
        load_path = os.path.join(self.save_dir, f"{filename}.pth")

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.net.load_state_dict( checkpoint['net_state_dict'] )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.tr_cases = checkpoint['tr_cases']    
    # ----------------------------------------------------------------------------------------------------
    def print_network(self):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.net.parameters(): num_params += param.numel()
        print(self.net)
        print('Network Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')               
    # ----------------------------------------------------------------------------------------------------
    def print_results(self, header, evalRes, N=8):
        txt = "\t" + header + ' '*(N-len(header)) + ':'
        txt_last = ''

        for key, val in evalRes.items(): 
            if key == 'mle_list': txt_last = f"{key}={val}" 
            else: txt += f"\t {key}={val:.6f}"
        
        print( f"{txt} \t {txt_last}")
####################################################################################################################################
