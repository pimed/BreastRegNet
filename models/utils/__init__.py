"""
    Author: Negar Golestani
    Created: August 2023
"""

import torch 
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import numpy as np


from .radImageNet import AddBlock, keras2pytorch


####################################################################################################################################
def load_pretrained_model(encoder_name, last_layer, attention=None, requires_grad_code=0, input_size=(1,3,224,224)):
    '''
        encoderInfo_str = '[EncoderInfo]-[LastLayer]-[RequireGrad]'
            - EncoderInfo:  resnet101, resnet101_cbam, radresnet50, vgg16, vgg16_cbam
            - LastLayer:    layer3, ...
            - RequireGrad:  0, 1, ... , "all"
    '''

    test_input = torch.rand(input_size)
    encoder = eval(encoder_name)()
    Nlayers = len(list(encoder.named_modules()))
    layers = OrderedDict()

    for i,(label,layer) in enumerate(encoder.named_children()):
        
        # Requires_grad
        if requires_grad_code == 'all': requires_grad = True                                    # make  all layers trainable
        else: requires_grad = False if i < Nlayers-int(requires_grad_code) else True            # freeze all layers except n=requires_grad_code last layers (NOTE: n=0 freezes all layers)
        for param in layer.parameters(): param.requires_grad = requires_grad                
        
        # Add attention
        if attention is not None:
            test_input = layer(test_input)
            output_channels = test_input.size()[1]    
            layer = nn.Sequential( OrderedDict( {label:layer, attention:eval(attention)(output_channels)} ))

        # Add layer
        layers[label] = layer  

        #  Crop
        if label == last_layer: break

    return nn.Sequential(layers) 
####################################################################################################################################
def vgg16(pretrained=True):   
    features_layers_name = [ 'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
                    'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
                    'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','pool3',
                    'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                    'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']    
    
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    layers = {n:l  for (n,l) in zip(features_layers_name, model.features.children())}
    layers = OrderedDict(
        layer1 = nn.Sequential(layers['conv1_1'],layers['relu1_1'],layers['conv1_2'],layers['relu1_2'], layers['pool1']),
        layer2 = nn.Sequential(layers['conv2_1'],layers['relu2_1'],layers['conv2_2'],layers['relu2_2'], layers['pool2']),
        layer3 = nn.Sequential(layers['conv3_1'],layers['relu3_1'],layers['conv3_2'],layers['relu3_2'], layers['conv3_3'],layers['relu3_3'], layers['pool3']),
        layer4 = nn.Sequential(layers['conv4_1'],layers['relu4_1'],layers['conv4_2'],layers['relu4_2'], layers['conv4_3'],layers['relu4_3'], layers['pool4']),
        layer5 = nn.Sequential(layers['conv5_1'],layers['relu5_1'],layers['conv5_2'],layers['relu5_2'], layers['conv5_3'],layers['relu5_3'], layers['pool5'])  ) 
    
    layers['avgpool'] = model.avgpool                                                                                                                                              
    layers['classifier'] = model.classifier                                                                                                                                              
    return nn.Sequential(layers)
####################################################################################################################################
def resnet50(pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    layers = {n:l for n,l in model.named_children()}
    layers = OrderedDict(layer0=nn.Sequential(layers.pop('conv1'),layers.pop('bn1'),layers.pop('relu'),layers.pop('maxpool')), **layers)
      
    return nn.Sequential(layers)
####################################################################################################################################
def resnet101(pretrained=True):
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weights=weights)
    layers = {n:l for n,l in model.named_children()}
    layers = OrderedDict(layer0=nn.Sequential(layers.pop('conv1'),layers.pop('bn1'),layers.pop('relu'),layers.pop('maxpool')), **layers)
      
    return nn.Sequential(layers)
####################################################################################################################################
def radresnet50():
    conveter = keras2pytorch('ResNet50')

    layers = OrderedDict()
    layers['layer0'] = conveter(['conv1_pad','conv1_conv','conv1_bn','conv1_relu','pool1_pad','pool1_pool'])
    layers['layer1'] = nn.Sequential(
        AddBlock(conveter(['conv2_block1_1_conv','conv2_block1_1_bn','conv2_block1_1_relu','conv2_block1_2_conv','conv2_block1_2_bn','conv2_block1_2_relu','conv2_block1_3_conv','conv2_block1_3_bn']),
                 conveter(['conv2_block1_0_conv','conv2_block1_0_bn']),
                 conveter(['conv2_block1_out']) 
                ),
        conveter(['conv2_block2_1_conv','conv2_block2_1_bn','conv2_block2_1_relu','conv2_block2_2_conv','conv2_block2_2_bn','conv2_block2_1_relu','conv2_block2_3_conv','conv2_block2_3_bn']),
        conveter(['conv2_block2_out']),
        conveter(['conv2_block3_1_conv','conv2_block3_1_bn','conv2_block3_1_relu','conv2_block3_2_conv','conv2_block3_2_bn','conv2_block3_2_relu','conv2_block3_3_conv','conv2_block3_3_bn']),
        conveter(['conv2_block3_out'])
    )
    layers['layer2'] = nn.Sequential(
        AddBlock(conveter(['conv3_block1_1_conv','conv3_block1_1_bn','conv3_block1_1_relu','conv3_block1_2_conv','conv3_block1_2_bn','conv3_block1_2_relu','conv3_block1_3_conv','conv3_block1_3_bn']),
                 conveter(['conv3_block1_0_conv','conv3_block1_0_bn']),
                 conveter(['conv3_block1_out']) 
                ), 
        conveter(['conv3_block2_1_conv','conv3_block2_1_bn','conv3_block2_1_relu','conv3_block2_2_conv','conv3_block2_2_bn','conv3_block2_2_relu','conv3_block2_3_conv','conv3_block2_3_bn']),
        conveter(['conv3_block2_out']),        
        conveter(['conv3_block3_1_conv','conv3_block3_1_bn','conv3_block3_1_relu','conv3_block3_2_conv','conv3_block3_2_bn','conv3_block3_2_relu','conv3_block3_3_conv','conv3_block3_3_bn']),
        conveter(['conv3_block3_out']),
        conveter(['conv3_block4_1_conv','conv3_block4_1_bn','conv3_block4_1_relu','conv3_block4_2_conv','conv3_block4_2_bn','conv3_block4_2_relu','conv3_block4_3_conv','conv3_block4_3_bn']),
        conveter(['conv3_block4_out'])
    )
    layers['layer3'] = nn.Sequential(
        AddBlock(conveter(['conv4_block1_1_conv','conv4_block1_1_bn','conv4_block1_1_relu','conv4_block1_2_conv','conv4_block1_2_bn','conv4_block1_2_relu', 'conv4_block1_3_conv','conv4_block1_3_bn']),
                 conveter(['conv4_block1_0_conv','conv4_block1_0_bn']),
                 conveter(['conv4_block1_out']) 
                ),
        conveter(['conv4_block2_1_conv','conv4_block2_1_bn','conv4_block2_1_relu','conv4_block2_2_conv','conv4_block2_2_bn','conv4_block2_2_relu','conv4_block2_3_conv','conv4_block2_3_bn']),
        conveter(['conv4_block2_out']),        
        conveter(['conv4_block3_1_conv','conv4_block3_1_bn','conv4_block3_1_relu','conv4_block3_2_conv','conv4_block3_2_bn','conv4_block3_2_relu','conv4_block3_3_conv','conv4_block3_3_bn']),
        conveter(['conv4_block3_out']),    
        conveter(['conv4_block4_1_conv','conv4_block4_1_bn','conv4_block4_1_relu','conv4_block4_2_conv','conv4_block4_2_bn','conv4_block4_2_relu','conv4_block4_3_conv','conv4_block4_3_bn']),
        conveter(['conv4_block4_out']),
        conveter(['conv4_block5_1_conv','conv4_block5_1_bn','conv4_block5_1_relu','conv4_block5_2_conv','conv4_block5_2_bn','conv4_block5_2_relu','conv4_block5_3_conv','conv4_block5_3_bn']),
        conveter(['conv4_block5_out']),
        conveter(['conv4_block6_1_conv','conv4_block6_1_bn','conv4_block6_1_relu','conv4_block6_2_conv','conv4_block6_2_bn','conv4_block6_2_relu','conv4_block6_3_conv','conv4_block6_3_bn']),
        conveter(['conv4_block6_out'])
    )
    layers['layer4'] = nn.Sequential(
        AddBlock(conveter(['conv5_block1_1_conv','conv5_block1_1_bn','conv5_block1_1_relu','conv5_block1_2_conv','conv5_block1_2_bn','conv5_block1_2_relu','conv5_block1_3_conv','conv5_block1_3_bn']),
                 conveter(['conv5_block1_0_conv','conv5_block1_0_bn']),
                 conveter(['conv5_block1_out'])
                ),
        conveter(['conv5_block2_1_conv','conv5_block2_1_bn','conv5_block2_1_relu','conv5_block2_2_conv','conv5_block2_2_bn','conv5_block2_2_relu','conv5_block2_3_conv','conv5_block2_3_bn']),
        conveter(['conv5_block2_out']),        
        conveter(['conv5_block3_1_conv','conv5_block3_1_bn','conv5_block3_1_relu','conv5_block3_2_conv','conv5_block3_2_bn','conv5_block3_2_relu','conv5_block3_3_conv','conv5_block3_3_bn']),
        conveter(['conv5_block3_out'])    
    )  

    return nn.Sequential(layers)
####################################################################################################################################


####################################################################################################################################
class Discriminator(nn.Module):
    def __init__(self, device='cpu'):
        super(Discriminator, self).__init__()
        self.device = device
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=3),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        ).to(device)
        self.fc = nn.Linear(in_features=128*7*7, out_features=1).to(device)

        # self.model = nn.Sequential(
        #                 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
        #                 nn.LeakyReLU(0.2),
        #                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        #                 nn.BatchNorm2d(64),
        #                 nn.LeakyReLU(0.2),    
        #                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        #                 nn.BatchNorm2d(128),
        #                 nn.LeakyReLU(0.2),   
        #                 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        #                 nn.BatchNorm2d(256),
        #                 nn.LeakyReLU(0.2) ).to(device)
        # self.fc = nn.Linear(in_features=256*14*14, out_features=1).to(device)
        
        self.sigmoid = nn.Sigmoid().to(device)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, x):
        x = self.model(x.to(self.device))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x    
####################################################################################################################################
class CAE(torch.nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),   
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=4)            
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=4, mode='nearest'),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(32, 3, kernel_size=3, padding=1),  
            torch.nn.Sigmoid()
        )           
    # ----------------------------------------------------------------------------------------------------
    def forward(self, x, max_val=255.0):
        z = self.encoder( x/max_val )
        y = self.decoder(z) * max_val
        return y
####################################################################################################################################
class TnfStabilizer(nn.Module):
    def __init__(self, geometric_model, device, theta_lr=.1):
        super().__init__()
        self.theta_lr = theta_lr
        self.device = device

        if geometric_model=='affine':
            self.identity = torch.tensor([1.0,0,0,0,1.0,0])           

        else:
            grid_size = int(geometric_model.split('_')[1])
            x = np.array([2*i/(grid_size-1)-1 for i in range(grid_size)]).reshape(-1,1)
            grd = np.array([*np.repeat(x,grid_size), *np.reshape([x]*grid_size, (1,-1))[0]]).reshape(-1,)
            self.identity = torch.tensor(grd, dtype=torch.float32)

            # self.identity = torch.tensor([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
            #                      -0.6,-0.6,-0.6,-0.6,-0.6,-0.6,
            #                      -0.2,-0.2,-0.2,-0.2,-0.2,-0.2,
            #                      0.2,0.2,0.2,0.2,0.2,0.2,
            #                      0.6,0.6,0.6,0.6,0.6,0.6,
            #                      1.0,1.0,1.0,1.0,1.0,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0,
            #                      -1.0,-0.6,-0.2,0.2,0.6,1.0])
    # ----------------------------------------------------------------------------------------------------
    def forward(self, theta):
        if self.theta_lr == 1: return theta

        theta = theta.view(-1, self.identity.shape[0])
        adjust = self.identity.repeat(theta.shape[0],1).to(self.device)
        theta = self.theta_lr*theta + adjust
        theta = theta.to(self.device)

        return theta                 
####################################################################################################################################
class CorrCNN(nn.Module):
    output_dim_dict = {'affine':6, 'tps':72}
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, geometric_model, feature_size, device):
        super().__init__()
        output_dim = self.output_dim_dict[geometric_model]
        b,c,h,w = feature_size
        kernel_size = 7 if h>7 else 5 

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(h*w, 64, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=kernel_size-2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) 
            ).to(device)
        
        # Get output size of conv network
        linInput_dim = 32 * (h-(2*kernel_size-2)+2) * (w-(2*kernel_size-2)+2)
        self.linear = nn.Linear(linInput_dim, output_dim).to(device)
        self.relu = nn.ReLU(inplace=True)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, feature_A, feature_B):        
        feature_A = nn.functional.normalize(feature_A, p=2, dim=(2,3))  # feature-wise normalization 
        feature_B = nn.functional.normalize(feature_B, p=2, dim=(2,3))

        x = correlation(feature_A, feature_B)
        x = self.relu(x)
        x = L2Norm(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
####################################################################################################################################
class CorrCNN2(nn.Module):
    output_dim_dict = {'affine':3, 'tps':72}
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, geometric_model, feature_size, device):
        super().__init__()
        output_dim = self.output_dim_dict[geometric_model]
        b,c,h,w = feature_size
        kernel_size = 7 if h>7 else 5 

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(h*w, 64, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=kernel_size-2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) 
            ).to(device)
        
        # Get output size of conv network
        linInput_dim = 32 * (h-(2*kernel_size-2)+2) * (w-(2*kernel_size-2)+2)
        self.linear = nn.Linear(linInput_dim, output_dim).to(device)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    # ----------------------------------------------------------------------------------------------------
    def forward(self, feature_A, feature_B):        
        feature_A = nn.functional.normalize(feature_A, p=2, dim=(2,3))  # feature-wise normalization 
        feature_B = nn.functional.normalize(feature_B, p=2, dim=(2,3))

        x = correlation(feature_A, feature_B)
        x = self.relu(x)
        x = L2Norm(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.tanh(x)
        return x    
####################################################################################################################################
def threshold(A, th=.95):
    b,c,h,w = A.size()
    Max = A.view((b,c,h*w)).max(dim=2)
    th_tensor = Max.values.unsqueeze(2).unsqueeze(3).expand_as(A) * th
    return ( th_tensor<A ) * A
####################################################################################################################################
def L2Norm(A, dim=1, epsilon=1e-12):
    # F.normalize(input, p=2, dim=1)
    mu = torch.pow(torch.sum(torch.pow(A,2), dim)+epsilon,0.5).unsqueeze(dim).expand_as(A)
    return torch.div(A, mu)
####################################################################################################################################
def correlation(A, B, epsilon=1e-12):
    # correlation         
    b,c,h,w = A.size()

    # reshape features for matrix multiplication
    A = A.view(b,c,h*w).transpose(1,2)
    B = B.view(b,c,h*w).transpose(1,2)
    
    # pearsonr
    mean_x = torch.mean(A, dim=2).unsqueeze(2).expand(b,h*w,c)
    mean_y = torch.mean(B, dim=2).unsqueeze(2).expand(b,h*w,c)

    xm = A.sub( mean_x )
    ym = B.sub( mean_y )

    var_x = torch.norm(xm, dim=2, p=2) + epsilon
    var_y = torch.norm(ym, dim=2, p=2) + epsilon

    r_num = xm.matmul( ym.transpose(1, 2) )
    r_den = (var_x * var_y).unsqueeze(2).expand(b,h*w,h*w)

    corr_pearson = r_num.div(r_den)
    corr_pearson = torch.abs(corr_pearson)
    corr = corr_pearson.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
    return corr
####################################################################################################################################
def multiplication(A, B):
    b,c,h,w = A.size()
    # reshape features for matrix multiplication
    A = A.transpose(2,3).contiguous().view(b,c,h*w)
    B = B.view(b,c,h*w).transpose(1,2)
    mul = torch.bmm(B,A)
    corr = mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
    return corr
####################################################################################################################################

