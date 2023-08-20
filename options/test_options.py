"""
    Author: Negar Golestani
    Created: August 2023
"""


import os

from .base_options import BaseOptions

####################################################################################################################################
class TestOptions(BaseOptions):
    # ----------------------------------------------------------------------------------------------------
    def initialize(self):
        super().initialize()
        self.parser.set_defaults(dataset_type='real')
        self.parser.add_argument('--names', required=True, type=str, help='name of models with "+" between them. (e.g., affine_v1+tps_v2"')
    # ----------------------------------------------------------------------------------------------------
    def set_extraOptions(self, opt):      
        opt_new = super().set_extraOptions(opt)
        opt_new.save_dir = os.path.join(opt_new.results_rootdir, f"{opt_new.names}_{opt_new.dataset}")     # Save directory
        opt_new.name_list = opt_new.names.split('+')   

        self.opt = opt_new
        return opt_new      
####################################################################################################################################

