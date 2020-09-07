import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.LEO.LEO_parts import *

class LEO(nn.Module):
    """docstring for Encoder"""
    def __init__(self, config):
        super(LEO, self).__init__()

        # Properties
        self._is_regression = False
        self._is_deterministic = config['is_deterministic']

        # Dimension for LEO part
        self.embed_size = config['embedding_size']
        self.hidden_size = config['hidden_size']

        # Hyper-parameter for LEO part
        self.dropout = config['dropout']
        self.kl_weight = config['kl_weight']
        self.encoder_panelty_weight = config['encoder_penalty_weight']
        self.orthogonality_penalty_weight = config['orthogonality_penalty_weight']

        # Update rule
        self.inner_update_step = config['inner_update_step']
        self.inner_lr = nn.Parameter(torch.FloatTensor([config['inner_lr_init']]))

        self.finetune_update_step = config['finetuning_update_step']
        self.finetune_lr = nn.Parameter(torch.FloatTensor([config['finetuning_lr_init']]))

        # LEO network
        self.model = LEO_network(self.embed_size, self.hidden_size, self.dropout, self._is_deterministic)

        # Model initialize
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, support_x, support_y, query_x):
        '''
            LEO
            
            Args:
                x_support : [task_size, n_way, k_shot, embed_size]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, embed_size]
                y_support : [task_size, n_way, k_shot, ] # not one-hot vector
            Returns:
                pred : [task_size, n_way, k_shot, num_classes] # 
        '''

        '''
        Support set
        '''

        # Shape
        task_size, n_way, k_shot, embed_size = query_x.size()

        # Encode a support_x
        latents, _ = self.model.encode(support_x)

        # Inner updates for latents
        for i in range(self.inner_update_step):
            # Grad
            latents.retain_grad()

            # Get weights
            classifier_weights = self.model.decode(latents)

            # train_loss in support set
            train_loss, _ = self.model.cal_target_loss(support_x, classifier_weights, support_y)

            # Get grad
            train_loss.backward(retain_graph=True)

            # Update latents
            latents = latents - (self.inner_lr * latents.grad)

        # Regenerate weights and prepare updating
        classifier_weights = self.model.decode(latents)
        

        # Finetune classifier_weights 
        for i in range(self.finetune_update_step):
            # Retain_grad
            classifier_weights.retain_grad()
            
            # train_loss in support set
            train_loss, train_acc = self.model.cal_target_loss(support_x, classifier_weights, support_y)

            # Get grad
            train_loss.backward(retain_graph=True)

            # Update classifier weights
            classifier_weights = classifier_weights - (self.finetune_lr * classifier_weights.grad)

        '''
        Query set
        '''
        logit_q = self.model.predict(query_x, classifier_weights) # [task_size * N_way * K_shot, num_classes (N_way)]
        logit_q = logit_q.view(task_size, n_way, k_shot, -1)

        # argmax operator
        pred_y = logit_q.argmax(dim=-1)

        return pred_y


    def meta_loss(self, support_x, support_y, query_x, query_y):
        '''
            LEO
            
            Args:
                x_support : [task_size, n_way, k_shot, embed_size]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, embed_size]
                y_query : [task_size, n_way, k_shot, ] # not one-hot vector
            Returns:
                loss (query_x, query_y) 
        '''

        # Encode a support_x / Regularizer 1 (encoder kl)
        latents, kl_div = self.model.encode(support_x)

        # Initialize a latent variable
        latents_init = latents

        # Inner updates for latents
        for i in range(self.inner_update_step):
            # Prepare Grad
            latents.retain_grad()

            # Get weights
            classifier_weights = self.model.decode(latents)

            # train_loss in support set
            train_loss, _ = self.model.cal_target_loss(support_x, classifier_weights, support_y)

            # Get grad
            train_loss.backward(retain_graph=True)

            # Update latents
            latents = latents - (self.inner_lr * latents.grad)

        # Regularizer 2 (latents)
        encoder_penalty = torch.mean((latents_init - latents) ** 2)

        # Regenerate weights and prepare updating
        classifier_weights = self.model.decode(latents)
        

        # Finetune classifier_weights 
        for i in range(self.finetune_update_step):
            # Prepare Grad
            classifier_weights.retain_grad()

            # train_loss in support set
            train_loss, train_acc = self.model.cal_target_loss(support_x, classifier_weights, support_y)

            # Get grad
            train_loss.backward(retain_graph=True)

            # Update classifier weights
            classifier_weights = classifier_weights - (self.finetune_lr * classifier_weights.grad)

        # Val loss
        val_loss, val_accuracy = self.model.cal_target_loss(query_x, classifier_weights, query_y)

        # Regularizer_3 : orthogonality_penalty
        orthogonality_penalty = self.orthogonality(list(self.model.decoder.parameters())[0])

        # Total loss
        if self._is_deterministic:
            total_loss = val_loss \
                + (self.encoder_panelty_weight * encoder_penalty) \
                + (self.orthogonality_penalty_weight * orthogonality_penalty)

        else:
            total_loss = val_loss + (self.kl_weight * kl_div) \
                + (self.encoder_panelty_weight * encoder_penalty) \
                + (self.orthogonality_penalty_weight * orthogonality_penalty)

        return total_loss, val_accuracy

    def orthogonality(self, weight):
        # For covariance matrix
        w2 = torch.mm(weight, weight.transpose(0, 1))
        wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20

        # Correlation matrix
        correlation_matrix = w2/ torch.mm(wn, wn.transpose(0, 1))
        assert correlation_matrix.size(0) == correlation_matrix.size(1)

        # Identity matrix
        I = torch.eye(correlation_matrix.size(0)).cuda()

        # Reularizer
        return torch.mean((correlation_matrix-I)**2)


if __name__ == "__main__":
    support_x = torch.Tensor(np.random.normal(size=(12,5,1,640))).float()
    support_y = torch.Tensor(np.random.randint(low=0, high=5, size=(12,5,1))).long()
    query_x = torch.Tensor(np.random.normal(size=(12,5,1,640))).float()
    support_y = torch.Tensor(np.random.randint(low=0, high=5, size=(12,5,1))).long()

    # Config
    config = yaml.load(open('/home/mgyukim/workspaces/MLwM/model/LEO/config.yml', 'r'), Loader=yaml.SafeLoader)
    config = config['miniImageNet']
    
    leo = LEO(config)

    query_y_hat = leo(support_x, support_y, query_x)    
    loss, acc = leo.meta_loss(support_x, support_y, query_x, support_y)

    print(query_y_hat.shape)
    print(loss)
