import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from .layer import *



class Abstract_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def _conv2d_img_size(self, maxpool_count=0):
        hidden_img_size = self._img_size

        for i in range(len(self._filter_sizes) + maxpool_count):
            hidden_img_size = math.floor((hidden_img_size + 2 * self._padding \
                - self._dilation * (self._kernel_size - 1) - 1) / self._stride + 1)
        return hidden_img_size

class Deterministic_Conv_Encoder(Abstract_Encoder):
    '''
        Deterministic_Conv_Encoder 
    '''

    def __init__(self, config, padding=0, stride=2, dilation=1, is_image_feature=True):
        '''
            NP
        '''

        super().__init__()

        #Input Dim
        self._img_size = config.get("img_size")
        self._input_channel = config.get("input_channel")
        self._filter_sizes = config.get("filter_size")
        self._kernel_size = config.get("kernel_size")
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._encoder_maxpool_count = config.get('encoder_maxpool_count', 0)


        # Output type
        self.is_image_feature = config.get('is_image_feature', True)
        
        # FC layer
        self._output_dim = config['encoder_output_dim'] if not self.is_image_feature \
            else config['encoder_output_dim_image']
        
        # FC layer
        self._hidden_img_size = self._conv2d_img_size(self._encoder_maxpool_count)

        # Embedding Layers
        self._layers = nn.ModuleList([])
        
        for i in range(len(self._filter_sizes)):
            if i == 0 :
                self._layers.append(nn.Conv2d(self._input_channel, self._filter_sizes[0], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
            elif i==len(self._filter_sizes)-1:
                if self._encoder_maxpool_count > 0:
                    self._layers.append(nn.MaxPool2d(2))
                self._layers.append(nn.Conv2d(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
            else:
                self._layers.append(nn.Conv2d(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
                
        # Last layer : FC
        self._last_layer = nn.Linear(self._filter_sizes[-1] * self._hidden_img_size**2, self._output_dim)
        
    
    def forward(self, inputs):
        '''
            Args:
                input : imamges (num_tasks, n_way, k_shot, filter_size img_size, img_size)

            Return:
                output : 
        '''
        # KL_loss = None (Initialize)
        kl_loss = None
        
        # Input shape
        num_task, n_way_k_shot, num_channel, img_size, img_size = inputs.shape

        # Reshaping (num_task, num_points) --> (num_task * num_points)
        inputs = inputs.view(-1, num_channel, img_size, img_size) #(num_task * n_way * k_shot, img_size, img_size)
        
        # Rename
        hidden = inputs
        
        # Conv Embedding (mu) for Meta learning
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Conv2d):
                hidden = F.relu(layer(hidden))
            else:
                hidden = layer(hidden)

        # Last layer (FC)
            # Flatten (num_task * n_way * k_shot ) --> (num_task, n_way, k_shot)
        hidden = hidden.view(num_task, n_way_k_shot, -1)  #(num_task, n_way, k_shot, output_dim)
        hidden = self._last_layer(hidden) #(num_task * num_points, output_dim)

        # Shape of img_size = sqrt(output_dim)
        output_dim = hidden.size(-1)
        img_size = int(math.sqrt(output_dim))

        # Reshape hidden to a image type
        if self.is_image_feature:
            hidden = hidden.view(num_task, n_way_k_shot, 1, img_size, -1)
        else:
            hidden = hidden.view(num_task, n_way_k_shot, -1)

    
        return hidden, kl_loss

class Stochastic_Conv_Encoder(Abstract_Encoder):
    '''
        Stochastic_Conv_Encoder 
    '''

    def __init__(self, config, padding=0, stride=2, dilation=1):
        '''
            NP
        '''

        super().__init__()

        #Input Dim
        self._img_size = config.get("img_size")
        self._input_channel = config.get("input_channel")
        self._filter_sizes = config.get("filter_size")
        self._kernel_size = config.get("kernel_size")
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._encoder_maxpool_count = config.get('encoder_maxpool_count', 0)


        # Output type
        self.is_image_feature = config.get('is_image_feature', True)
        
        # FC layer
        self._output_dim = config['encoder_output_dim'] if not self.is_image_feature \
            else config['encoder_output_dim_image']
        
        # FC layer
        self._hidden_img_size = self._conv2d_img_size(self._encoder_maxpool_count)

        # Embedding Layers
        self._layers = nn.ModuleList([])

        for i in range(len(self._filter_sizes)):
            if i == 0 :
                layer = Stochastic_Conv2D(self._input_channel, self._filter_sizes[0], self._kernel_size, stride=self._stride)
            elif i==len(self._filter_sizes)-1:
                if self._encoder_maxpool_count > 0:
                    self._layers.append(nn.MaxPool2d(2))
                layer = Stochastic_Conv2D(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, stride=self._stride)
            else:
                layer = Stochastic_Conv2D(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, stride=self._stride)
            #Append
            self._layers.append(layer)
            
        # Last layer : FC
        self._last_layer = Stochastic_FC(self._filter_sizes[-1] * self._hidden_img_size**2, self._output_dim)
        

    def forward(self, inputs):
        '''
            Args:
                input : imamges (num_tasks, num_points (way * shot), img_size, img_size)

            Return:
                output : 
        '''
        # Input shape
        num_task, n_way_k_shot, num_channel, img_size, img_size = inputs.shape

        # Reshaping (num_task, n_way, k_shot) --> (num_task * n_way * k_shot)
        inputs = inputs.view(-1, num_channel, img_size, img_size)
        
        # Rename
        hidden = inputs

        # kl_loss
        kl_loss = 0

        # Stochastic Conv Embedding for Meta learning
        for i, layer in enumerate(self._layers):
            if hasattr(layer,'kl_loss'):
                hidden = F.relu(layer(hidden))
                kl_loss += layer.kl_loss()
            else:
                hidden = layer(hidden)

        # Last layer (FC)
            # Flatten (num_task * n_way * k_shot ) --> (num_task, n_way, k_shot)
        hidden = hidden.view(num_task, n_way_k_shot, -1)  #(num_task, n_way, k_shot, output_dim)
        hidden = self._last_layer(hidden) #(num_task * num_points, output_dim)

        # KL loss
        kl_loss += self._last_layer.kl_loss()

        # Shape of img_size = sqrt(output_dim)
        output_dim = hidden.size(-1)
        img_size = int(math.sqrt(output_dim))

        # Reshape hidden to a image type
        if self.is_image_feature:
            hidden = hidden.view(num_task, n_way_k_shot, 1, img_size, -1)
        else:
            hidden = hidden.view(num_task, n_way_k_shot, -1)

        return hidden, kl_loss

class Conv_Reparameterization_Encoder(Abstract_Encoder):
    '''
        Stochastic_Conv_Encoder 
    '''

    def __init__(self, config, padding=0, stride=2, dilation=1, is_image_feature=True):
        '''
            NP
        '''

        super().__init__()

        #Input Dim
        self._img_size = config.get("img_size")
        self._input_channel = config.get("input_channel")
        self._filter_sizes = config.get("filter_size")

        self._kernel_size = config.get("kernel_size")
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._encoder_maxpool_count = config.get('encoder_maxpool_count', 0)

        # Output type
        self.is_image_feature = config.get('is_image_feature', True)

        # FC layer
        self._hidden_img_size = self._conv2d_img_size(self._encoder_maxpool_count)
        self._output_dim = config['encoder_output_dim'] if not self.is_image_feature \
            else config['encoder_output_dim_image']

        # Embedding Layers
        self._layers = nn.ModuleList([])

        for i in range(len(self._filter_sizes)):
            if i == 0 :
                self._layers.append(nn.Conv2d(self._input_channel, self._filter_sizes[0], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
            elif i==len(self._filter_sizes)-1:
                if self._encoder_maxpool_count > 0:
                    self._layers.append(nn.MaxPool2d(2))
                self._layers.append(nn.Conv2d(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
            else:
                self._layers.append(nn.Conv2d(self._filter_sizes[i-1], self._filter_sizes[i], self._kernel_size, \
                    stride=self._stride, padding=self._padding, dilation=self._dilation, groups=1))
            

        # Last layer : FC
        self._last_layer_mu = nn.Linear(self._filter_sizes[-1] * self._hidden_img_size**2, self._output_dim)
        self._last_layer_rho = nn.Linear(self._filter_sizes[-1] * self._hidden_img_size**2, self._output_dim)
        

    def forward(self, inputs):
        '''
            Args:
                input : imamges (num_tasks, n_way, k_shot, num_channel, img_size, img_size)

            Return:
                output : encoded_img, kl_loss
        '''
        # Input shape
        num_task, n_way_k_shot, num_channel, img_size, img_size = inputs.shape

        # Reshaping (num_task, num_points) --> (num_task * num_points)
        inputs = inputs.view(-1, num_channel, img_size, img_size) #(num_task * n_way * k_shot, img_size, img_size)

        # Rename
        hidden = inputs
        
        # Conv Embedding (mu) for Meta learning
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Conv2d):
                hidden = F.relu(layer(hidden))
            else:
                hidden = layer(hidden)
            

        # Flatten (num_task * n_way * k_shot, num_channel, imgsize, imgsize) --> (num_task, n_way, k_shot, reminging_dim)
        hidden = hidden.view(num_task, n_way_k_shot, -1)
        
        hidden_mu = self._last_layer_mu(hidden) #(num_task, n_way, k_shot, output_dim)
        hidden_rho = self._last_layer_rho(hidden) #(num_task, n_way, k_shot, output_dim)

        # Distribution and sampling
        dist_hidden = torch.distributions.Normal(hidden_mu, torch.log1p(torch.exp(hidden_rho))) #(num_task * num_points, output_dim)
        dist_prior = torch.distributions.Normal(0, 1)

        # KL_loss
        kl_loss = torch.distributions.kl.kl_divergence(dist_hidden, dist_prior).sum()
        hidden_reparameterization = dist_hidden.rsample() #(num_task * num_points, output_dim)

        # Shape of img_size = sqrt(output_dim)
        output_dim = hidden_reparameterization.size(-1)
        img_size = int(math.sqrt(output_dim))

        # Reshape hidden to a image type
        if self.is_image_feature:
            hidden_reparameterization = hidden_reparameterization.view(num_task, n_way_k_shot, 1, img_size, -1)
        else:
            hidden_reparameterization = hidden_reparameterization.view(num_task, n_way_k_shot, -1)

        return hidden_reparameterization, kl_loss

class Deterministic_FC_Encoder(Abstract_Encoder):
    '''
        Deterministic_Conv_Encoder 
    '''

    def __init__(self, config):
        '''
            NP
        '''

        super().__init__()

        #Input Dim
        self._feature_dim = config[0]
        
        # FC layer
        self._hidden_dim = config[1]
        self._output_dim = config[2]
        self._layer_count = config[3]

        # Embedding Layers
        self._layers = nn.ModuleList([])
        
        for i in range(len(self._filter_sizes)):
            if i == 0 :
                self._layers.append(nn.Linear(self._feature_dim, self._hidden_dim))
            if i == (self._layer_count - 1):
                self._layers.append(nn.Linear(self._hidden_dim, self._output_dim))
            else:
                self._layers.append(nn.Linear(self._hidden_dim, self._hidden_dim))

    def forward(self, inputs):
        '''
            Args:
                input : imamges (num_tasks, n_way, k_shot, feature_dim)

            Return:
                output : 
        '''
        # Input shape
        num_task, n_way, k_shot, feature_dim  = inputs.shape

        # Reshaping (num_task, num_points) --> (num_task * num_points)
        inputs = inputs.view(-1, feature_dim) #(num_task * n_way * k_shot, img_size, img_size)
        
        # Rename
        hidden = inputs
        
        # Conv Embedding (mu) for Meta learning
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))

        # Last layer (FC)
            # Flatten (num_task * n_way * k_shot ) --> (num_task, n_way, k_shot)
        hidden = hidden.view(num_task, n_way, k_shot, -1)  #(num_task, n_way, k_shot, output_dim)
        hidden = self._last_layer(hidden) #(num_task * num_points, output_dim)

        # Shape of img_size = sqrt(output_dim)
        output_dim = hidden.size(-1)
        
        # KL_loss = None
        kl_loss = None

        return hidden, kl_loss
                
class Stochastic_FC_Encoder(Abstract_Encoder):
    '''
        Deterministic_Conv_Encoder 
    '''

    def __init__(self, config, is_image_feature=True):
        '''
            NP
        '''

        super().__init__()

        #Input Dim
        self._feature_dim = config[0]

        # Output type
        self.is_image_feature = is_image_feature
        
        # FC layer
        self._hidden_dim = config[1]
        self._output_dim = config[2]
        self._layer_count = config[3]

        # Embedding Layers
        self._layers = nn.ModuleList([])
        
        for i in range(self._layer_count):
            # Initialize
            layer = None
            
            if i == 0 :
                layer = Stochastic_FC(self._feature_dim, self._hidden_dim)
            elif i == (self._layer_count - 1):
                layer = Stochastic_FC(self._hidden_dim, self._output_dim)
            else:
                layer = Stochastic_FC(self._hidden_dim, self._hidden_dim)

            if layer is not None:
                self._layers.append(layer)
            else:
                NotImplementedError
                

    def forward(self, inputs):
        '''
            Args:
                input : imamges (num_tasks, n_way, k_shot, feature_dim)

            Return:
                output : 
        '''
        # Input shape
        num_task, n_way, k_shot, feature_dim  = inputs.shape

        # Reshaping (num_task, num_points) --> (num_task * num_points)
        inputs = inputs.view(-1, feature_dim) #(num_task * n_way * k_shot, img_size, img_size)
        
        # Rename
        hidden = inputs

        # kl_loss
        kl_loss = 0
        
        # Conv Embedding (mu) for Meta learning
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))
            kl_loss += layer.kl_loss()

        # Shape of img_size = sqrt(output_dim)
        output_dim = hidden.size(-1)
        img_size = int(math.sqrt(output_dim))

        # Reshape hidden to a image type
        if self.is_image_feature:
            hidden = hidden.view(num_task, n_way, k_shot, 1, img_size, -1)
        else:
            hidden = hidden.view(num_task, n_way, k_shot, -1)

        return hidden, kl_loss



if __name__ == "__main__":
    # DEBUG
    DEBUG = True

    # Parameter
    IMG_SIZE = 128

    # Data path
    get_data_dir = '/home/mgyukim/Data/rotate_resize/Dataset'
    data = ['train_data_2.pkl', 'val_data_2.pkl']

    # Data loader
    pose_regression_filepath = '/home/mgyukim/Data/rotate_resize/Dataset'
    dataset = meta_pose_regression_dataset(5, 5, 5, pose_regression_filepath, types='non_mutual_exclusive')
    dataset.reset_episode()

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    
    
    if DEBUG:
        for i, data in enumerate(train_dataloader):

            # Difine the datasize
            support_x, support_y, query_x, query_y = data

            # Print Shape
            print('support set image shape : ', support_x.shape)

            break

    # Call
        # model 1(BBB) : IMG_SIZE, input_channel, output_dim (FC), [# of channels], kernel_size
    model_1 = Stochastic_Conv_Encoder(ENCODER_CONFIG[1])
        # model 2(VAE) : IMG_SIZE, input_channel, output_dim (FC), [# of channels], kernel_size
    model_2 = Conv_Reparameterization_Encoder(ENCODER_CONFIG[1])
        # model 3 (Deterministic) : IMG_size, input_channel, output_dim, [# of channels], kernel_size
    model_3 = Deterministic_Conv_Encoder(ENCODER_CONFIG[1])

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data

        # Type casting
        support_x = support_x.type(torch.FloatTensor)
        support_y = support_y.type(torch.FloatTensor)
        query_x = query_x.type(torch.FloatTensor)
        query_y = query_y.type(torch.FloatTensor)


    
        print(support_x[0][4][4])
        print(support_x.dtype)
        print(support_x.max())
        print(support_y.max())

        encoded_img, kl_loss_1 = model_1(support_x)
        encoded_img_2, kl_loss_2 = model_2(support_x)
        encoded_img_3, _ = model_3(support_x)

        print(encoded_img.shape)
        print(encoded_img_2.shape)
        print(encoded_img_3.shape)

        print(kl_loss_1.shape)
        print(kl_loss_2.shape)


        print(kl_loss_1)
        print(kl_loss_2)


        break

    

    
        



    
