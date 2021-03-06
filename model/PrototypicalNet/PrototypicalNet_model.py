
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PrototypicalNet.PrototypicalNet_parts import *


class PrototypeNet_embedded(nn.Module):
    """
    Classifier whose weights are generated by averaging feature_dim w.r.t kshots:
    
    """
    def __init__(self, n_way, config):
        '''
        Args : 
            n_way : provide a hidden_size of syntehtic_grad_net
            config : config
        '''
        
        
        super(PrototypeNet_embedded, self).__init__()

        self.n_way = n_way
        self.feature_dim = config['feature_dim']
        
        # init_net lambda(d_t^l)
            # Prototype net for initial theta
        self.Prototype_net = Prototype_net()
            # Linear mapping for linear mapping from original theta

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
        x_transformed = F.normalize(x_transformed, p=2, dim=x_transformed.dim()-1, eps=1e-12) #[batch_size, n_way * k_shot, feature_dim]
        weights = F.normalize(weights, p=2, dim=weights.dim()-1, eps=1e-12) #[batch_size, n_way, feature_dim]
        weights = weights.transpose(1,2) #[batch_size, feature_dim, n_way]

        # logits [task_size, n_way * k_shot, n_way]
        logits =  torch.bmm(x_transformed, weights)

        # Reshape : logits
        logits = logits.view(task_size, n_way, k_shot, -1) # [batch_size, n_way, k_shot, n_way (num_classes)]

        return logits


    def get_prototype(self, support_x_transformed, support_y_one_hot):
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
        prototypes = self.Prototype_net(support_x_transformed, support_y_one_hot) # [batch_size, n_way, dim]

        return prototypes


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
        prototypes = self.get_prototype(support_x_transformed, labels_supp_1hot)

        # calculate logits
            # [batch_size, n_way, k_shot, feature_dim]  
        logits = self.apply_classification_weights(query_x_transformed, prototypes)
            
        return logits, prototypes

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
        # criterion
        criterion_1 = F.cross_entropy
        
        # Get shape
        task_size, n_way, k_shot_support, feature_dim = support_x_transformed.size()
        _, _, k_shot_query, _ = query_x_transformed.size()

        # Evaluate logits
        logits, prototypes = self.forward_logits(support_x_transformed, support_y, query_x_transformed)
            #[batch_size, n_way, k_shot, n_way]

        # Reshape
        logits = logits.view(-1, n_way) #[batch_size * n_way * k_shot, n_way]
        query_y_flatten = query_y.view(-1) #[batch_size * n_way * k_shot,]

        # Loss : Evaluate loss for query_y and predicted value
        meta_loss = criterion_1(logits, query_y_flatten)

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




