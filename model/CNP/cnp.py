import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .parts.encoder import Deterministic_Conv_Encoder, \
    Conv_Reparameterization_Encoder, \
    Stochastic_Conv_Encoder

import yaml
from torch.utils.data import Dataset, DataLoader


class Encoder(nn.Module):
    '''
        Deterministic Encoder
    '''

    def __init__(self, input_dim, hidden_dim, rep_dim):
        '''
            Np deterministic encoder
            
            Args:
                input_dim : x_dim + y_dim
                hidden_dim : hidden_dim
                rep_dim : representation dim
                layer_sizes : the array of each lyaer size in encoding MLP
        '''

        super().__init__()

        # Dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim

        # MLP embedding architectures
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.rep_dim)
        )

        self.encoder_r_z = nn.Linear(self.rep_dim, self.rep_dim)

    def forward(self, support_x, support_y, target_x=None):
        '''
            Ecoding the input into representation using latent encoder

            Args:
                context_x: [batch_size, the number of observation, x_size(dimension)] 
                context_y : [batch_size, the number of observation, y_size(dimension)] 

            Returns:
                representation : [batch_size, the nmber of observation, num_lantents]
        '''

        # Concatenate x and y along the filter axises
            # Technique
        encoder_input = torch.cat((support_x, support_y), dim=-1) # [batch_size, the number of points, x_size + y_size]

        # Shape
        task_size, _, filter_size = tuple(encoder_input.shape)

        # Input
        hidden = encoder_input.view((-1, filter_size))

        # MLP embedidng for NP
        hidden = self.encoder(hidden)
        
        
        # Reshaping
        hidden = hidden.view((task_size, -1, self.rep_dim)) # [batch_size, the number of point, the last element in the list]

        # Conditional Neural Processes
            # Aggregation of representation
        rep = hidden.mean(dim=1)

        # Sampling
        rep = self.encoder_r_z(rep)

        return rep

class Decoder(nn.Module):
    '''
        (A)NP decoder
    '''

    def __init__(self, input_dim, rep_dim, hidden_dim, output_dim):
        '''
            Args:
                input_dim : x_dim + latent_dim
                layer_sizes : the array of each layer size in encoding NP
        '''

        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Decoder
        self.decoder_y = nn.Sequential(
            nn.Linear(self.input_dim + self.rep_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # Distribution
        self.hidden_to_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden_to_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    
    def forward(self, target_x, stochastic_rep, deterministic_rep=None):
        '''
            Decoders the individual targets

            Args:
                representation : [batch_size, the number of points, rep_dim (or None)]
                target_x : [batch_size, the number of points, x_dim(dim)]

            Returns:
                dist : multivariate gaussian dist. Sample from this distribution has [batch_size, the number of points, y_size(dim)]
                mu : mean of multivariate distribution [batch_size, the number of points, y_size(dim)]
                sigma : std of multivariate didstribution [batch_size, the number of points, y_size(dim)]
        '''
        # Concatenate target_x and representation        
        hidden = torch.cat((stochastic_rep, target_x), dim = -1)
        
        # Shape
        batch_size, _, filter_size = tuple(hidden.shape)

        # Input
        hidden = hidden.view(-1, filter_size)

        # Exceptional Treatement
        assert filter_size == (self.input_dim + self.rep_dim), \
            "You must match the dimension of input_dim and representations + target_x"
        
        # Decoder
        hidden = self.decoder_y(hidden)
            
        # Forwarding mean and std
            # [batch_size*num_data_points, num_lantets]
        mu = self.hidden_to_mu(hidden) 
        log_sigma = self.hidden_to_sigma(hidden)

        # Reshaping mu and log_sigma
        mu = mu.view(batch_size, -1, self.output_dim)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma.view(batch_size, -1, self.output_dim))

        return mu, sigma

class CNP(nn.Module):
    def __init__(self, cnp_config, initializer=None, \
        is_regression=False, is_image_feature=True):

        super().__init__()

        # Loss
        self.loss_fn = F.mse_loss
        
        # Conv Encoder
        self.encoder_type = cnp_config.get('encoder_type', None)
        self.cnp_config = cnp_config
        self._is_regression = is_regression
        self._is_image_feature = cnp_config.get('is_image_feature', False)
        self._is_kl_loss = cnp_config.get('is_kl_loss', False) \
            if not self.encoder_type == "deterministic" else False
        self._beta_kl = cnp_config['beta_kl']

        # Dim
        self.dim_w = cnp_config['dim_w']
        self.dim_embed_y = self.dim_w // cnp_config['dim_embed_y_divisor']
        self.dim_r = cnp_config['dim_r']
        self.dim_z = cnp_config['dim_z']
        self.dim_y = cnp_config['dim_y']
        self.hidden_r = cnp_config['hidden_r']
        self.hidden_g = cnp_config['hidden_g']
        self.hidden_d = cnp_config['hidden_d']

        # Encoder_z
        if self.encoder_type == 'deterministic':
            self.encoder = \
                Deterministic_Conv_Encoder(self.cnp_config)

        elif self.encoder_type == 'VAE':
            self.encoder = \
                Conv_Reparameterization_Encoder(self.cnp_config)
        
        elif self.encoder_type == 'BBB':
            self.encoder = \
                Stochastic_Conv_Encoder(self.cnp_config)

        # Encoder_y
        self.encoder_y = nn.Sequential(
            nn.Linear(self.dim_y, self.dim_embed_y)
        )

        # Encoder_rep
        self.encoder_rep = Encoder(self.dim_w + self.dim_embed_y, self.hidden_r, self.dim_z)

        # Decoder
        self.decoder = Decoder(self.dim_w, self.dim_z, self.hidden_d, self.dim_y)


    def forward(self, x_support, y_support, x_query, is_kl_loss=False):
        '''
            Model Agnostic Meta Learning with an encoder
            
            Args:
                x_support : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, channel, height, width]

            Returns:
                pred : [task_size, n_way, k_shot, ] # 
        '''
        # Get shape
        task_size, n_way, k_shot, channel, height, width = x_query.size()
        _, _, _, dim_y = y_support.size()

        # Reshape
        x_support = x_support.view(task_size, -1, channel, height, width)
        x_query = x_support.view(task_size, -1, channel, height, width)
        y_support = y_support.view(task_size, -1, dim_y)
        
        # Encode x_support and x_query
        w_support, kl_loss_encoder = self.encoder(x_support)
        w_query, _ = self.encoder(x_query)

        # Encoded y 
        embed_y_support = self.encoder_y(y_support)

        # Encoder_representation
        rep = self.encoder_rep(w_support, embed_y_support)

        # 뻥튀기
            #[batch_size, the number of points, latent_dim]
        rep = torch.unsqueeze(rep, dim=1).repeat(1, n_way * k_shot, 1)  

        # Decoding
        mu, sigma = self.decoder(w_query, rep)

        # Distribution
        p_y_dist = torch.distributions.Normal(mu, sigma)

        # Predict
        pred_y = p_y_dist.rsample()

        if is_kl_loss:
            return pred_y, kl_loss_encoder

        else:
            return pred_y


    def meta_loss(self, x_support, y_support, x_query, y_query, is_hessian=True):
        '''
            Model Agnostic Meta Learning
            
            Args:
                x_support : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, channel, height, width]
                y_query : [task_size, n_way, k_shot, ] # not one-hot vector

            Returns:
                loss : Loss for the meta parameters
        '''

        '''
        _, _, k_shot_support, _, _, _ = x_support.size()
        task_size, n_way, k_shot_query, channel_count, height, width = x_query.size()
        '''
        # Garbage
        losses_list = None, None
        
        # Get shape
        task_size, n_way, k_shot, channel, height, width = x_query.size()

        # Forward
        pred_y, kl_loss_encoder = self.forward(x_support, y_support, x_query, is_kl_loss=True)

        # Reshape
        pred_y = pred_y.view(task_size, n_way, k_shot, -1)
        
        # Forward
        sum_task_loss = self.loss_fn(pred_y, y_query)

        # Averaging
        mse_loss = sum_task_loss /  task_size

        # KL loss
        if self._is_kl_loss:
            beta_kl = self._beta_kl * kl_loss_encoder
            loss = mse_loss + beta_kl
        else:
            loss = mse_loss
        
        return loss, mse_loss, losses_list


# For debug
def set_config_encoder(config, encoder_type, encoder_output_dim):
    '''
        Set last fc layer should be setted as img_size * img_size

        Args :
            config : encoder configs
            img_size : original img_size

        Return :
            config : encoder configs list
    '''

    # Encoder types
    if encoder_type == "deterministic" or \
        encoder_type == "VAE" or \
            encoder_type == "BBB" or \
                encoder_type == "BBB_FC":

        config[0] = encoder_type
    
    else :
        NotImplementedError

    properties = config[1]
    properties[2] = encoder_output_dim

    config[1] = properties

    return config


if __name__ == '__main__':
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

    # Import config
    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/cnp_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
    config = config['Pose_regression']

    ENCODER_CONFIG = set_config_encoder(config['ENCODER_CONFIG'], \
            config['encoder_type'], config['encoder_output_dim'])

    cnp_config = config


    # model : IMG_SIZE, input_channel, output_dim (FC), [# of channels], kernel_size
    model = CNP(ENCODER_CONFIG, cnp_config, is_regression=True)
        

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

        pred_y = model(support_x, support_y, query_x)
        loss = model.meta_loss(support_x, support_y, query_x, query_y)
        
        print(pred_y.shape)
        print(loss)
        

        break

        


        