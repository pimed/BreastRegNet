"""
    Author: Negar Golestani
    Created: August 2023
"""

import os
from sys import exit 
from shutil import rmtree

from .base_options import BaseOptions

####################################################################################################################################
class TrainOptions(BaseOptions):
    # ----------------------------------------------------------------------------------------------------
    def initialize(self):
        super().initialize()
        self.parser.add_argument('-n', '--name', type=str, required=True, help='name of the experiment')
        self.parser.add_argument('--image_w', type=int, default=224, help='image width')
        self.parser.add_argument('--image_h', type=int, default=224, help='image height')
        self.parser.add_argument('--rot_range', type=int, default=20, help='range of rotation angle for synthetic dataset (-/+ degree)')
        self.parser.add_argument('--move_range', type=float, default=.1, help='range of image shifting coefficients for synthetic dataset (ratio of movement to image size)')
        self.parser.add_argument('--scale_range', type=float, default=.1, help='range of image scaling coefficients for synthetic dataset') # NOTE: 1-scale_range:1+scale_range
        self.parser.add_argument('--shear_range', type=float, default=.05, help='range of image shearing coefficients for synthetic dataset (ratio of movement to image size)')
        # Network parameters
        self.parser.add_argument('--model', type=str, default='regModel', help='network name [regDualModel|regModel|segModel]')
        self.parser.add_argument('--network', type=str, default='regCorrDualNet-vgg16-layer4-1', help='network name and info')
        # self.parser.add_argument('--encoder', type=str, default='resnet101', help='encoder description [ resnet101 | resnet101-CBAM | resnet101-BAM | radresnet50 ] ')
        self.parser.add_argument('--geometric_model', type=str, default='affine', help='geometric model to be regressed at output [affine | tps | rigidPuzzle]')
        self.parser.add_argument('--loss', type=str, default='mse', help='type of loss [tnfGrid|tnf|mse|dice|lde]')       # NOTE: for more than one loss function use "+". e.g.: 2*tnfGrid+1*ssd or alternate_tnfGrid+ssd
        # Training parameters 
        self.parser.add_argument('--theta_lr', type=float, default=.1, help='learning rate for theta [ 1 means no stabalizing]')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.95, help='gamma for scheduler')
        self.parser.add_argument('--step_size', type=int, default=1, help='step size for scheduler')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--num_workers', type=int, default=0, help='number of worker of multi-process data loader')
        self.parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')   
        # self.parser.add_argument('--grid_size', type=int, default=3, help='size of grid')       
        self.parser.add_argument('--random_t_tps', type=float, default=.3, help='TPS transformation parameter')       
        # Validation parameters 
        self.parser.add_argument('--val_type', type=str, default='5-foldFixed', help='type of cross  validation [none |0.2-split | 5-fold |5-foldFixed]') # None means no validation
    # ----------------------------------------------------------------------------------------------------
    def set_extraOptions(self, opt):
        opt_new = super().set_extraOptions(opt)
        opt_new.save_dir = os.path.join(opt_new.checkpoints_rootdir, opt_new.name)     # Save directory        

        # opt_new.loss = opt_new.loss.split('+')
        opt_new.validate = False if opt_new.val_type.lower() == 'none' else True 

        self.opt = opt_new
        return opt_new
    # ----------------------------------------------------------------------------------------------------
    def save(self, opt, save_dir):
        # If directory exists 
        if os.path.exists(save_dir):
            while True:
                ans = input(f'\n > {opt.name} already exists. Do you want to overwrite it? [y/n] \n')
                if ans.lower() == 'n': exit()
                elif ans.lower() == 'y': rmtree(save_dir); break 

        super().save(opt, save_dir)
####################################################################################################################################
