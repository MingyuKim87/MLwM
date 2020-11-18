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