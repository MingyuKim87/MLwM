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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SIB.SIB_parts import *


class SIB(nn.Module):
    """
    Classifier whose weights are generated dynamically from synthetic gradient descent:
    Objective: E_{q(w | d_t^l, x_t)}[ -log p(y | feat(x), w) ] + KL( q(w|...) || p(w) )

    Note: we use a simple parameterization
        - q(w | d_t^l, x_t) = Dirac_Delta(w - theta^k),
          theta^k = synthetic_gradient_descent(x_t, theta^0)
          theta^0 = init_net(d_t)
        - p(w) = zero-mean Gaussian and implemented by weight decay
        - p(y=k | feature(x), w) = prototypical network

    :param int n_way: number of categories in a task/episode.
    :param int feature_dim: feature dimension of the input feature.
    :param int q_steps : number of synthetic gradient steps to obtain q(w | d_t^l, x_t).
    """
    def __init__(self, n_way, config):
        '''
        Args : 
            n_way : provide a hidden_size of syntehtic_grad_net
            config : config
        '''
        
        
        super(SIB, self).__init__()

        self.n_way = n_way
        self.feature_dim = config['feature_dim']
        self.update_step = config['inner_update_step']
        self.lr = config['inner_lr']
        self.coefficient_synthetic_grad_loss = config['coefficient_synthetic_grad_loss']

        # bias & scale of classifier p(y | x, theta)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # init_net lambda(d_t^l)
            # Prototype net for initial theta
        self.Prototype_net = Prototype_net()
            # Linear mapping for linear mapping from original theta
        self.Linear_mapping_net = Linear_Mapping(self.feature_dim)

        # grad_net (aka decoupled network interface) phi(x_t)
            # Synthetic gradient networks
            # Translate to Logit for class [batch, n_way * k_shot, n_way] --> [batch, n_way * k_shot, n_way]
        self.synthetic_grad_net = Synthetic_grad_linear(self.n_way, hidden_size=self.n_way*8)

        # setting
        self._is_regression = False


    def apply_classification_weights(self, x_transformed, weights):
        """
        Computing logits for classification

        Args:
            x_transformed : feature of query_x [batch_size, n_way, k_shot, feature_dim]
            weights : prototype_weight [batch_size, n_way, dim]

        Returns : 
            logits : [batch_size, n_way, k_shot, feature_dim]
        """
        # Reshape
        task_size, n_way, k_shot, feature_dim = x_transformed.size()
        x_transformed = x_transformed.view(task_size, -1, feature_dim)

        # Normalizing
        x_transformed = F.normalize(x_transformed, p=2, dim=x_transformed.dim()-1, eps=1e-12)
        weights = F.normalize(weights, p=2, dim=weights.dim()-1, eps=1e-12)

        # logits [task_size, n_way * k_shot, n_way]
        logits = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0,
                                                    x_transformed, weights.transpose(1,2))

        # Reshape : logits
        logits = logits.view(task_size, n_way, k_shot, -1) # [batch_size, n_way, k_shot, n_way (num_classes)]

        return logits


    def init_theta(self, support_x_transformed, support_y_one_hot):
        """
        Compute theta^0 from support set using classwise feature averaging.
            (Prototype)

        Args:
            support_x_transformed : [batch_size, n_way, k_shot, dim]
            support_y_one_hot : [batch_size, n_way, k_shot, n_way]
        Returns : 
            theta : [bathc_size, n_way, dim]
        """
        # Reshaping
        task_size, n_way, k_shot, feature_dim = support_x_transformed.size()
        _, _, _, num_classes = support_y_one_hot.size()
        support_x_transformed = support_x_transformed.view(task_size, -1, feature_dim)
        support_y_one_hot = support_y_one_hot.view(task_size, -1, num_classes)
        
        # Calculate prototype
        theta = self.Prototype_net(support_x_transformed, support_y_one_hot) # [batch_size, n_way, dim]

        theta = theta.view(-1, feature_dim)
        theta = self.Linear_mapping_net(theta) # linear transform of weight for each feature differently
        theta = theta.view(-1, n_way, feature_dim) # [task_size, n_way, dim]
        return theta

    def refine_theta(self, theta, query_x_transformed, lr=1e-3):
        """
        Compute theta^k using synthetic gradient descent on query_x.

        Args : 
            theta : [batch_size, n_way, dim]
            query_x_transformed : [batch_size, n_way, k_shot, dim]

        Returns :
            theta_k : [batch_size, n_way, dim]
        """
        # Reshaping
        task_size, n_way, k_shot, feature_dim = query_x_transformed.size()
        total_num_points = task_size * n_way * k_shot

        for _ in range(self.update_step):
            # calculate logits by using 'weight == theta' (dirac delta func.)
            logits = self.apply_classification_weights(query_x_transformed, theta) #[batch_size, n_way, k_shot, n_way]
            
            # Reshaping
            logits = logits.view(total_num_points, -1) # [batch_size * n_way * k_shot, n_way]

            # Synthetic gradient for logits
            synthetic_grad_logits = self.synthetic_grad_net(logits) # [batch_size * n_way * k_shot, n_way] --> [batch_size * n_way * k_shot, n_way]
            
            # [batch_size, n_way, dim], [0] means the first element in output tuples. 
            grad_theta = torch.autograd.grad([logits], [theta],
                                       grad_outputs=[synthetic_grad_logits],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0] 
            

            # perform synthetic GD
            theta = theta - lr * grad_theta

        return theta

    def get_classification_weights(self, support_x_transformed, support_y_one_hot, \
        query_x_transformed, lr):
        """
        Obtain weights for the query_x using support_x, support_y and query_x.
            support_x, support_y --> self.init_theta
            query_x --> self.refine_theta (by syntehtic grad)

        Args:
            support_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            support_y_one_hot : [batch_size, n_way, k_shot, n_way]
            query_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            lr : scalar

        Returns:
            weights : dirac delta func (theta) [batch_size, n_way, n_dim]
        """

        # Normalizing
        support_x_transformed = F.normalize(support_x_transformed, p=2, \
            dim=support_x_transformed.dim()-1, eps=1e-12)

        theta = self.init_theta(support_x_transformed, support_y_one_hot)
        theta_k = self.refine_theta(theta, query_x_transformed, lr)

        # Sampling weights from theta_k having dirac delta pdf.
        weights = theta_k #[task_size, n_way, dim]

        return weights


    def forward(self, support_x_transformed, support_y, query_x_transformed):
        """
        Return predicted value (same dimension of query_y).

        Args:
            support_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            support_y : [batch_size, n_way, k_shot]
            query_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            lr : scalar

        Returns :
            logits : [batch_size, n_way, k_shot, feature_dim]
        """
        # Get shapes
        task_size, n_way, k_shot_support, feature_dim = support_x_transformed.size()
        _, _, k_shot_query, _ = query_x_transformed.size()
        
        
        # Forward
            # [batch_size * n_way * k_shot, n_way]
        logits, _ = self.forward_logits(support_x_transformed, support_y, query_x_transformed)

        # Reshape
        logits = logits.view(task_size, n_way, k_shot_query, -1)

        # argmax operator
            # [batch_size, n_way, k_shot, n_wqy]
        pred_y = logits.argmax(dim=-1)

        return pred_y


    def forward_logits(self, support_x_transformed, support_y, query_x_transformed):
        """
        Compute classification scores.

        Args:
            support_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            support_y : [batch_size, n_way, k_shot]
            query_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            lr : scalar

        Returns :
            logits : [batch_size, n_way, k_shot, feature_dim]
        """
        # Transform target to one-hot valued target
        labels_supp_1hot = label_to_1hot(support_y, self.n_way)

        # weights = theta_k by dirac delta pdf.
        weights = self.get_classification_weights(support_x_transformed, labels_supp_1hot, \
            query_x_transformed, self.lr)

        # calculate logits
        logits = self.apply_classification_weights(query_x_transformed, weights)
            # [batch_size, n_way, k_shot, feature_dim]

        return logits, weights 


    def meta_loss(self, support_x_transformed, support_y, query_x_transformed, query_y):
        """
        Cross entropy loss

        Args:
            support_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            support_y : [batch_size, n_way, k_shot]
            query_x_transformed : [batch_size, n_way, k_shot, feature_dim]
            query_y : [batch_size, n_way, k_shot]
            lr : scalar

        Returns :
            loss : scalar (float)
            accuracy : scalar (float)
        """
        
        # For hook, move to grad --> grad_nonleaf 
            # It does not give impact on Autograd graph of "backward"

        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g
            h = v.register_hook(hook)
            return h
        
        
        # criterion
        criterion_1 = F.cross_entropy
        criterion_2 = F.mse_loss

        # Get shape
        task_size, n_way, k_shot_support, feature_dim = support_x_transformed.size()
        _, _, k_shot_query, _ = query_x_transformed.size()

        # Evaluate logits
        logits, weights = self.forward_logits(support_x_transformed, support_y, query_x_transformed)
            #[batch_size, n_way, k_shot, n_way]

        # Reshape
        logits = logits.view(-1, n_way) #[batch_size * n_way * k_shot, n_way]
        query_y_flatten = query_y.view(-1) #[batch_size * n_way * k_shot,]

        # Loss 1 : Evaluate loss for query_y and predicted value
        loss_classification = criterion_1(logits, query_y_flatten)

        # Loss 2 : Synthetic gradient
        synthetic_grad_logits = self.synthetic_grad_net(logits)
        handle = require_nonleaf_grad(logits)

            # eval grad of logits
        loss_classification.backward(retain_graph=True)

            # remove handle
        handle.remove()

            # true graident of logits
        grad_logits = logits.grad_nonleaf.detach()
        #grad_logits = torch.autograd.grad([loss_classification], [logits])[0].detach()

        loss_gradients = criterion_2(synthetic_grad_logits, grad_logits)

        # meta loss
        meta_loss = loss_classification + \
            (self.coefficient_synthetic_grad_loss * loss_gradients)
        
        # Acuuracy
        pred = logits.argmax(dim=-1)
        corr = (pred == query_y_flatten).sum()
        total_num = torch.ones_like(pred).sum()

        accuracy = corr.float() / total_num.float()

        return meta_loss, accuracy



        






        

        


if __name__ == "__main__":
    '''
    net = ClassifierSIB(nKall=64, nKnovel=5, nFeat=512, q_steps=3)
    net = net.cuda()

    features_supp = torch.rand((8, 5 * 1, 512)).cuda()
    features_query = torch.rand((8, 5 * 15, 512)).cuda()
    labels_supp = torch.randint(5, (8, 5 * 1)).cuda()
    lr = 1e-3

    cls_scores = net(features_supp, labels_supp, features_query, lr)
    print(cls_scores.size())
    '''

    a = torch.ones(2,2, requires_grad=True)
    b = 2*torch.ones(2,2, requires_grad=False)

    c = a * b

    d = c.sum()
    
    

    grad = torch.autograd.grad(d, c)

    

    #d.backward(retain_graph=True)    
    

    
    print(grad)
    print(grad[0])
    

    '''
    def require_nonleaf_grad(v):
        def hook(g):
            v.grad_nonleaf = g
        h = v.register_hook(hook)
        return h
    
    handle = require_nonleaf_grad(c)
    '''
    

    
    d.backward(retain_graph=True)

    print(d.shape)

    #handle.remove()

    print(a.grad_fn)
    print(c.grad_fn)
    print(c.grad)
    print(c.grad_nonleaf.detach())
    #print(a.grad_nonleaf)




