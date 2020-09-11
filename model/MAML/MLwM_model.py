import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from model.MAML.maml_meta import Meta
from model.MAML.meta_sgd import MetaSGD

from model.MAML.Part.encoder import Deterministic_Conv_Encoder, \
    Stochastic_Conv_Encoder, Conv_Reparameterization_Encoder, Stochastic_FC_Encoder

from torch.utils.data import DataLoader


class MLwM(nn.Module):
    def __init__(self, encoder_config, maml_config, update_lr, update_step, initializer=None, \
        is_regression=False, is_image_feature=True, is_kl_loss=False, beta_kl=1, is_meta_sgd=False):
        super().__init__()
        
        self.encoder_type = encoder_config[0]
        self.encoder_property = encoder_config[1]
        self.maml_config = maml_config
        self._is_regression = is_regression
        self._is_image_feature = is_image_feature
        self._is_kl_loss = is_kl_loss
        self._beta_kl = beta_kl
        self._is_meta_sgd = is_meta_sgd

        if encoder_config[0] == 'deterministic':
            self.encoder = \
                Deterministic_Conv_Encoder(self.encoder_property)

        elif self.encoder_type == 'VAE':
            self.encoder = \
                Conv_Reparameterization_Encoder(self.encoder_property)
        
        elif self.encoder_type == 'BBB':
            self.encoder = \
                Stochastic_Conv_Encoder(self.encoder_property)

        elif self.encoder_type == 'BBB_FC':
            self.encoder = \
                Stochastic_FC_Encoder(self.encoder_property)
        else:
            NotImplementedError

        # choose 'update_lr' can be learned or not
        if self._is_meta_sgd:
            self.maml = MetaSGD(self.maml_config, update_lr, update_step, initializer=None, is_regression=self._is_regression, is_image_feature=self._is_image_feature )
        else:
            self.maml = Meta(self.maml_config, update_lr, update_step, initializer=None, is_regression=self._is_regression, is_image_feature=self._is_image_feature )

    def forward(self, x_support, y_support, x_query, is_hessian=True):
        '''
            Model Agnostic Meta Learning with an encoder
            
            Args:
                x_support : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, ] # not one-hot vector
            Returns:
                pred : [task_size, n_way, k_shot, num_classes] # 
        '''
        '''
        _, _, k_shot_support, _, _, _ = x_support.size()
        task_size, n_way, k_shot_query, channel_count, height, width = x_query.size()
        '''

        # Encode x_support and x_query
        encoded_x_support, _ = self.encoder(x_support)
        encoded_x_query, _ = self.encoder(x_query)

        # Forward by MAML
        pred_y_stack = self.maml.forward(encoded_x_support, y_support, encoded_x_query, is_hessian) #[task_size, n_way, k_shot_query]

        return pred_y_stack

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

        # Encode x_support and x_query
        encoded_x_support, kl_loss_support = self.encoder(x_support)
        encoded_x_query, kl_loss_query = self.encoder(x_query)

        
        # Forward by MAML
        maml_loss, criterion = self.maml.meta_loss(encoded_x_support, y_support, encoded_x_query, y_query, is_hessian) #[task_size, n_way, k_shot_query]

        # kl_loss and total loss
        if self.encoder_type != "deterministic" and self._is_kl_loss:
                kl_loss = (kl_loss_support + kl_loss_query) / 2.
                total_loss = maml_loss + (self._beta_kl * kl_loss)
        else:
            total_loss = maml_loss

        return total_loss, criterion


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

    # Concatenate "fc" for the config file
    MAML_CONFIG = set_config(MAML_CONFIG, 1, 28)

    
    # model : IMG_SIZE, input_channel, output_dim (FC), [# of channels], kernel_size
    model = MLwM(ENCODER_CONFIG, MAML_CONFIG, update_lr = 0.002, update_step=5, is_regression=True)
        

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
        loss, accuracy = model.meta_loss(support_x, support_y, query_x, query_y)
        
        print(pred_y.shape)
        print(loss)
        print(accuracy)

        break











            