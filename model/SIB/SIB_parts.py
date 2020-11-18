# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def label_to_1hot(label, K):
    # Get shapes
    task_size, n_way, k_shot = label.size()
    
    # Empty container for return value
    labels_one_hot = torch.repeat_interleave(torch.zeros_like(label).unsqueeze(-1), n_way, dim=-1) 
    

    # For scatter
        # Shaping [total_point, 1]
    labels = label.unsqueeze(-1)
    labels = labels.view(-1, 1)
        # Shaping [total_point, n_way (num_classes)]
    labels_one_hot = labels_one_hot.view(-1, n_way)

    # One hot vectors
    labels_one_hot = labels_one_hot.scatter_(1, labels, 1)

    #Type cast
    labels_one_hot = labels_one_hot.float()

    # Reshaping
    labels_one_hot = labels_one_hot.view(task_size, n_way, k_shot, -1)
    
    return labels_one_hot

def fc_layer(layer_count, input_size, output_size, hidden_size=None, \
    is_last_layer_activated=False, is_batch_norm=False,  bias=False):
        
    modules = []

    for i in range(layer_count):
        if layer_count == 1:
            layer = nn.Linear(input_size, output_size, bias=bias)
            modules.append(layer)
        else:
            if i == layer_count -1 and is_last_layer_activated:
                layer = nn.Linear(hidden_size, output_size, bias=bias)
                activation = nn.ReLU()
                modules.append(layer)
                modules.append(activation)

            elif i == layer_count -1 and not is_last_layer_activated:
                layer = nn.Linear(hidden_size, output_size, bias=bias)
                modules.append(layer)
                
            elif i == 0:
                layer = nn.Linear(input_size, hidden_size, bias=bias)
                activation = nn.ReLU()

                modules.append(layer)
                modules.append(activation)

                if is_batch_norm:
                    batch_norm = nn.BatchNorm1d(hidden_size)
                    modules.append(batch_norm)
                
            else:
                layer = nn.Linear(hidden_size, hidden_size,bias=bias)
                activation = nn.ReLU()

                modules.append(layer)
                modules.append(activation)

                if is_batch_norm:
                    batch_norm = nn.BatchNorm1d(hidden_size)
                    modules.append(batch_norm)

    layer_sequence = nn.Sequential(*modules)
            
    return layer_sequence


class Synthetic_grad_linear(nn.Module):
    '''
        Generate synthetic gradients with clsScore

    '''
    def __init__(self, input_dims, layer_count=3, hidden_size=1024):
        super(Synthetic_grad_linear, self).__init__()
        '''
        self.layer1 = nn.Sequential(
                      nn.Linear(input_dims, hidden_size),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_size)
                      )
        self.layer2 = nn.Sequential(
                      nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_size)
                      )
        self.layer3 = nn.Linear(hidden_size, input_dims)
        '''
        self.synthetic_grad_net = fc_layer(layer_count, input_dims, input_dims, hidden_size, False, True, True)

    def forward(self, x):
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        '''

        out = self.synthetic_grad_net(x)

        return out


class Linear_Mapping(nn.Module):
    def __init__(self, feature_dim, bias=False):
        '''
            linear operator with weights 
                : For improving theta (weights) by vectorized weights (same dimension for task_size * n_way * k_shot, weight) 
                : Linear Mapping from original theta to new theta
        '''
        super(Linear_Mapping, self).__init__()
        weight = torch.FloatTensor(feature_dim).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(feature_dim).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        # Exceptional Treatment
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))

        # Element wise multiplication
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            # Element wise add
            out = out + self.bias.expand_as(out)
            
        return out


class Prototype_net(nn.Module):
    def __init__(self):
        '''
            Provide weights by multiplying feature and one_hot valued target 
                (calculate prototypes)
        '''
        super(Prototype_net, self).__init__()

    def forward(self, features_train, labels_train):
        '''
            Args:
                Feature_train : [B, n_way * k_shot, feature_dim]
                labels_train : [B, n_way * k_shot, n_way]

            Returns : 
                weights : [B, n_way, feature_dim]

        '''
        labels_train_transposed = labels_train.transpose(1,2) # [B, n_way, n_way * k_shot]
        # BMM : [B, n_way, n_way * k_shot] * [B, n_way * k_shot, feature_dim] --> [B, n_way, feature_dim]
        weights = torch.bmm(labels_train_transposed, features_train)

        # Normalize : 더해준 개수만큼 나눠줌
        weights = weights.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weights))

        return weights