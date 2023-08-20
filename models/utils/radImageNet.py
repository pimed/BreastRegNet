'''
    Code adapted from: https://dthiagarajan.github.io/technical_blog/draft/pytorch/hooks/2020/03/18/Dynamic-UNet-and-PyTorch-Hooks.html

'''

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from options.config import MODELS_ROOTDIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u



####################################################################################################################################
# RadImageNet
####################################################################################################################################
class keras2pytorch(object):    
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, model_name, models_dir=MODELS_ROOTDIR):
        self.models_dir = models_dir
        self.layersConfig_dict = self.load_layersConfig(model_name)
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, layerName_list):
        layers = list()

        for layer_name in layerName_list:
            config = self.layersConfig_dict[layer_name]
            # ----------------------
            if config['type'] =='ZeroPadding2D':
                (top_pad, bottom_pad), (left_pad, right_pad) = config['padding']
                layer = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))            
            # ----------------------
            elif config['type'] == 'Conv2D':
                layer = nn.Conv2d( 
                    in_channels = config['input_shape'][-1],
                    out_channels = config['filters'],
                    kernel_size = config['kernel_size'],
                    stride = config['strides'],
                    padding = config['padding'],
                    dilation = config['dilation_rate'],
                    bias = config['use_bias']  )
                # weights
                w, b = config['weights']
                layer.weight.data = torch.Tensor(np.transpose(w))
                layer.bias.data = torch.Tensor(np.transpose(b))           
            # ----------------------
            elif config['type'] =='BatchNormalization':   
                layer = nn.BatchNorm2d(config['input_shape'][-1], affine=True, momentum=config['momentum'], eps=config['epsilon'])
                # weights
                gamma, beta, mean, variance = config['weights']
                layer.weight.data = torch.Tensor(np.array(gamma))
                layer.bias.data = torch.Tensor(np.array(beta))
                layer.running_mean.data = torch.Tensor(np.array(mean))
                layer.running_var.data = torch.Tensor(np.array(variance))
            # ----------------------
            elif config['type'] =='Activation':             
                layer = nn.ReLU()  # all activations are relu 
            # ----------------------
            elif config['type'] =='MaxPooling2D':
                layer = nn.MaxPool2d(
                    kernel_size = config['pool_size'],
                    stride = config['strides'],
                    padding = config['padding'] )
            # ----------------------
            layers.append( (layer_name, layer)  )

        return nn.Sequential(OrderedDict(layers))
    # ----------------------------------------------------------------------------------------------------
    def load_layersConfig(self, model_name):
        load_path = os.path.join(self.models_dir, f"RadImageNet-{model_name}_notop.pickle")

        # File does not exist
        if not os.path.exists(load_path): 
            doCreate_str = input(f"'{load_path}' does not exist. If you want to create the file press [Y]:")
            if doCreate_str.lower() == 'y': self.save_layersConfig(self.models_dir, model_name)
            else: return
        
        # load pickle file
        with open(load_path, 'rb') as handle: layersConfig_dict = pickle.load(handle)
        return layersConfig_dict
    # ----------------------------------------------------------------------------------------------------
    def save_layersConfig(self, model_name):
        '''
            Keras to pickle (.h5 to .pickle) 
            NOTE: have to save configs due to runtime memory errors 
            'name' example: RadImageNet-ResNet50_notop
        '''
        from tensorflow.keras.models import load_model

        # Load keras model
        model_keras = load_model(os.path.join(self.models_dir, f"RadImageNet-{model_name}_notop.h5"), compile=False)

        # Get layers config
        layersConfig_dict = dict()
        for layer in model_keras.layers:
            config = layer.get_config()
            config['type'] = layer.__class__.__name__
            config['input_shape'] = layer.input_shape
            if 'padding' in config and config['padding'] == 'valid': config['padding'] = 0
            config['weights'] = layer.get_weights()
            
            name = config.pop('name')
            layersConfig_dict[name] = config

        # save
        save_path = os.path.join(self.models_dir, f"RadImageNet-{model_name}_notop.pickle")
        with open(save_path, 'wb') as handle:
            pickle.dump(layersConfig_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
####################################################################################################################################
class AddBlock(nn.Module):
    def __init__(self, sub1, sub2, out):
        super().__init__()
        self.sub1 = sub1
        self.sub2 = sub2
        self.out = out
    # ----------------------------------------------------------------------------------------------------
    def forward(self, x):
        return self.out( self.sub1(x)+self.sub2(x) )
####################################################################################################################################

