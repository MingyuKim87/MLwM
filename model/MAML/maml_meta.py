import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


from model.MAML.Part.maml_learner import Learner
from copy import deepcopy #all components should be 

DEBUG = True

# Set Config
config = list()
config.append(('conv2d', [128, 3, 32, 32, 1, 0])) #[ch_out, ch_in, size, size, stride, padding]
config.append(('flatten', []))
config.append(('fc', [1, 128]))


def gradient_descent(grad, param, lr, is_hessian=True):
        new_weight = param - (lr * grad)

        if not is_hessian:
            new_weight = new_weight.detach()

        return new_weight

class Meta(nn.Module):
    '''
        MAML algorithm and model class
    '''

    def __init__(self, config, update_lr, update_step, initializer=None, is_regression=False, is_image_feature=True):
        super(Meta, self).__init__()

        # Parameters
        self.update_lr = update_lr
        self.update_step = update_step
        
        # Model
        self.net = Learner(config, initializer=initializer)
        self._is_regression=is_regression
        self._is_image_feature = is_image_feature

    
    def set_update_step(self, update_step):
        self.update_step = update_step

    def clip_grad_by_norm_(self, grad, max_norm):
        '''
            Cliping gradients by max_norm
            refers to "torch.utils.clip_grad"
        '''

        total_norm = 0
        counter = 0

        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1

        total_norm = torch.sqrt(total_norm)

        clip_coef = max_norm / (total_norm + 1e+6)

        if clip_coef < 1 :
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_support, y_support, x_query, is_hessian=True, is_adaptation=True):
        '''
            Model Agnostic Meta Learning
            
            Args:
                x_support : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, ] # not one-hot vector
            Returns:
                pred : [task_size, n_way, k_shot, num_classes] # 
        '''
        if self._is_image_feature:
            _, _, k_shot_support, _, _, _ = x_support.size()
            task_size, n_way, k_shot_query, channel_count, height, width = x_query.size()

            # Reshape x and y
            x_support = x_support.view(task_size, -1, channel_count, height, width)
            x_query = x_query.view(task_size, -1, channel_count, height, width)
        else:
            _, _, k_shot_support, _,  = x_support.size()
            task_size, n_way, k_shot_query, feature_dim = x_query.size()

            # Reshape x and y
            x_support = x_support.view(task_size, -1, feature_dim)
            x_query = x_query.view(task_size, -1, feature_dim)

        # Depending on the problem type, target dimension should be changed. 
        if self._is_regression:
            # For mathcing a dimension, [n_way * k_shot, 1] by evaluating F.mse_loss 
            # It requires same same logtis and target [n_way * k_shot, 1]
            y_support = y_support.view(task_size, -1, 1)
        else:
            # By loss func, F.cross_entropy requires a target shape : [n_way * k_shot] 
            y_support = y_support.view(task_size, -1)

        # Pred Container
        pred_y_list = []
        
        for i in range(task_size):
            # Run the i-th task and compute loss for k = 0 
                # Forward (bn_training)
            logits = self.net(x_support[i], vars=None, bn_training=True)
        
            if not is_adaptation:
                if not self._is_regression:
                    pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[n_way * k_shot_query, ]
                    pred_y = pred_q.view(n_way, k_shot_query) #[n_way, k_shot_query]
                    pred_y_list.append(pred_y)

                else:
                    pred_y_list.append(logit_q)

            else:
                if self._is_regression:
                    task_loss = F.mse_loss(logits, y_support[i])
                else:
                    task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')

                # Parameters
                parameters = self.net.parameters()

                # Gradient
                grads = torch.autograd.grad(task_loss, parameters)
                
                # Initializing the container of task-specific parameters
                task_parameter = []

                # Inner Update
                for j, (grad, param) in enumerate(zip(grads, parameters)):
                    new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                    task_parameter.append(new_weight)

                # By using the query set, we assess loss and accuracy with meta-parameters
                with torch.no_grad():
                    # Evaluated by meta parameter
                    logit_q = self.net(x_query[i], self.net.parameters(), bn_training=True) #[n_way * k_shot, classess]
                    if not self._is_regression:
                        pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[n_way * k_shot,]

                    # Evaluated by first updated parameter
                    logit_q = self.net(x_query[i], task_parameter, bn_training=True)
                    if not self._is_regression:
                        pred_q = F.softmax(logit_q, dim=1).argmax(dim=1)
                    
                for k in range(1, self.update_step):
                    # Run the i-th task and compute loss for k = 1
                    # Forward (bn_training)
                    logits = self.net(x_support[i], vars=task_parameter, bn_training=True)
                    
                    if not self._is_regression:
                        task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')
                    else:
                        task_loss = F.mse_loss(logits, y_support[i])

                    # Gradient
                    grads = torch.autograd.grad(task_loss, task_parameter)

                    # Inner update
                    for j, (grad, param) in enumerate(zip(grads, task_parameter)):
                        new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                        task_parameter[j] = new_weight

                    with torch.no_grad():
                        # Assessment
                        logit_q = self.net(x_query[i], task_parameter, bn_training=True) #[n_way * k_shot_query, num_class]
                        if not self._is_regression:
                            pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[n_way * k_shot_query, ]
                            pred_y = pred_q.view(n_way, k_shot_query) #[n_way, k_shot_query]

                if not self._is_regression:
                    pred_y_list.append(pred_y)
                else:
                    pred_y_list.append(logit_q)
                    
        # Reshape
        pred_y_stack = torch.stack(pred_y_list, dim=0) #[task_size, n_way, k_shot_query], type=torch.long
        
        return pred_y_stack


    def meta_loss(self, task_x_support, task_y_support, task_x_query, task_y_query, is_hessian=True):
        '''
            Model Agnostic Meta Learning
            
            Args:
                task_x_support : [task_size, n_way, k_shot, channel, height, width]
                task_y_support : [task_size, n_way, k_shot, ] # not one-hot vector
                task_x_query : [task_size, n_way, k_shot, channel, height, width]
                task_y_query : [task_size, n_way, k_shot, ] # not one-hot vector

            Returns:
                loss : Loss for the meta parameters
        '''
        if self._is_image_feature:
            _, _, k_shot_support, _, _, _ = task_x_support.size()
            task_size, n_way, k_shot_query, channel_count, height, width = task_x_query.size()

            # Reshape x and y
            x_support = task_x_support.view(task_size, -1, channel_count, height, width)
            x_query = task_x_query.view(task_size, -1, channel_count, height, width)
        else:
            _, _, k_shot_support, _,  = task_x_support.size()
            task_size, n_way, k_shot_query, feature_dim = task_x_query.size()

            # Reshape x and y
            x_support = task_x_support.view(task_size, -1, feature_dim)
            x_query = task_x_query.view(task_size, -1, feature_dim)
        
        # Depending on the problem type, target dimension should be changed. 
        if self._is_regression:
            # For mathcing a dimension, [n_way * k_shot, 1] by evaluating F.mse_loss 
            # It requires same same logtis and target [n_way * k_shot, 1]
            y_support = task_y_support.view(task_size, -1, 1)
            y_query = task_y_query.view(task_size, -1, 1)
        else:
            # By loss func, F.cross_entropy requires a target shape : [n_way * k_shot] 
            y_support = task_y_support.view(task_size, -1)
            y_query = task_y_query.view(task_size, -1)

        # Initializing Containers for inner updates
        losses_q = [0] * (self.update_step + 1)
        corrects = [0] * (self.update_step + 1)
        
        for i in range(task_size):
            # Run the i-th task and compute loss for k = 0 
                # Forward (bn_training)
            logits = self.net(x_support[i], vars=None, bn_training=True)            
            if self._is_regression:
                task_loss = F.mse_loss(logits, y_support[i])
            else:
                task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')

            # Parameters
            parameters = self.net.parameters()

            # Gradient
            grads = torch.autograd.grad(task_loss, parameters)
            
            # Initializing the container of task-specific parameters
            task_parameter = []

            # Inner Update
            for j, (grad, param) in enumerate(zip(grads, parameters)):
                new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                task_parameter.append(new_weight)

            # By using the query set, we assess loss and accuracy with meta-parameters
            with torch.no_grad():
                # Evaluated by meta-parameter
                logit_q = self.net(x_query[i], self.net.parameters(), bn_training=True) #[num_points, classess]
                pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[num_points,]

                # Evaluating loss
                if self._is_regression:
                    loss_q = F.mse_loss(logit_q, y_support[i])
                else:
                    loss_q = F.cross_entropy(logit_q, y_support[i], reduction='sum')
                
                # Evaluation 
                losses_q[0] += loss_q / task_size
                correct = torch.eq(pred_q, y_query[i]).sum().item() if not self._is_regression else None
                corrects[0] += correct / (task_size * n_way * k_shot_query) if not self._is_regression else 0

                # Evaluated by first updated parameter
                logit_q = self.net(x_query[i], task_parameter, bn_training=True) #[num_points, classess]
                pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[num_points,]

                # Evaluating loss
                if self._is_regression:
                    loss_q = F.mse_loss(logit_q, y_query[i])
                else:
                    loss_q = F.cross_entropy(logit_q, y_query[i], reduction='sum')
                
                losses_q[1] += loss_q / task_size
                correct = torch.eq(pred_q, y_query[i]).sum().item() if not self._is_regression else None
                corrects[1] += correct / (task_size * n_way * k_shot_query) if not self._is_regression else 0

            for k in range(1, self.update_step):
                # Run the i-th task and compute loss for k = 1 
                # Forward (bn_training)
                logits = self.net(x_support[i], vars=task_parameter, bn_training=True)
                if self._is_regression:
                    task_loss = F.mse_loss(logits, y_support[i])
                else:
                    task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')

                # Gradient
                grads = torch.autograd.grad(task_loss, task_parameter)

                # Inner update
                for j, (grad, param) in enumerate(zip(grads, task_parameter)):
                    new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                    task_parameter[j] = new_weight

                # Assessment
                logit_q = self.net(x_query[i], task_parameter, bn_training=True)
                pred_q = F.softmax(logit_q, dim=1).argmax(dim=1)

                # Evaluating loss
                if self._is_regression:
                    loss_q = F.mse_loss(logit_q, y_query[i])
                else:
                    loss_q = F.cross_entropy(logit_q, y_query[i], reduction='sum')

                    with torch.no_grad():
                        # correct : scalar (summation of n_way * k_shot_query)
                        correct = torch.eq(pred_q, y_query[i]).sum().item() if not self._is_regression else None
                        corrects[k+1] += correct / (task_size * n_way * k_shot_query) if not self._is_regression else 0

                losses_q[k+1] += loss_q / task_size

        if not self._is_regression:
            # criterion = accuracy
            criterion = corrects
        else:
            # criterion = mse_loss
            criterion = losses_q[-1]

        return losses_q[-1], criterion[-1] if not self._is_regression else losses_q[-1], losses_q


    def get_embedded_vector_forward(self, x_support, y_support, x_query, is_hessian=True, is_adaptation=False):
        '''
            Model Agnostic Meta Learning
            
            Args:
                x_support : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, , ] # not one-hot vector
                x_query : [task_size, n_way, k_shot, channel, height, width]
                y_support : [task_size, n_way, k_shot, ] # not one-hot vector
            Returns:
                embedded_vectors : [task_size, n_way, k_shot, embedded_dims] # 
        '''
        
        if self._is_image_feature:
            _, _, k_shot_support, _, _, _ = x_support.size()
            task_size, n_way, k_shot_query, channel_count, height, width = x_query.size()

            # Reshape x and y
            x_support = x_support.view(task_size, -1, channel_count, height, width)
            x_query = x_query.view(task_size, -1, channel_count, height, width)

        else:
            _, _, k_shot_support, _,  = x_support.size()
            task_size, n_way, k_shot_query, feature_dim = x_query.size()

            # Reshape x and y
            x_support = x_support.view(task_size, -1, feature_dim)
            x_query = x_query.view(task_size, -1, feature_dim)

        # Depending on the problem type, target dimension should be changed. 
        if self._is_regression:
            # For mathcing a dimension, [n_way * k_shot, 1] by evaluating F.mse_loss 
            # It requires same same logtis and target [n_way * k_shot, 1]
            y_support = y_support.view(task_size, -1, 1)
        else:
            # By loss func, F.cross_entropy requires a target shape : [n_way * k_shot] 
            y_support = y_support.view(task_size, -1)

        # Get Parameters
        parameters = self.net.parameters()
        
        # Pred Container
        embedded_vector_list = []

        if not is_adaptation:
            for i in range(task_size):
                embedded_vector = self.net.get_embedded_vector(x_query[i], parameters)
                embedded_vector = embedded_vector.view(n_way, k_shot_query, -1)
                embedded_vector_list.append(embedded_vector)

            # Reshape
            embedded_vector_stack = torch.stack(embedded_vector_list, dim=0) #[task_size, n_way, k_shot_query], type=torch.long

            return embedded_vector_stack

        else:
            for i in range(task_size):
                # Run the i-th task and compute loss for k = 0 
                    # Forward (bn_training)
                logits = self.net(x_support[i], vars=None, bn_training=True)
            
                if self._is_regression:
                    task_loss = F.mse_loss(logits, y_support[i])
                else:
                    task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')

                # Parameters
                parameters = self.net.parameters()

                # Gradient
                grads = torch.autograd.grad(task_loss, parameters)
                
                # Initializing the container of task-specific parameters
                task_parameter = []

                # Inner Update
                for j, (grad, param) in enumerate(zip(grads, parameters)):
                    new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                    task_parameter.append(new_weight)

                # By using the query set, we assess loss and accuracy with meta-parameters
                with torch.no_grad():
                    # Evaluated by meta parameter
                    logit_q = self.net(x_query[i], self.net.parameters(), bn_training=True) #[n_way * k_shot, classess]
                    if not self._is_regression:
                        pred_q = F.softmax(logit_q, dim=1).argmax(dim=1) #[n_way * k_shot,]

                    # Evaluated by first updated parameter
                    logit_q = self.net(x_query[i], task_parameter, bn_training=True)
                    if not self._is_regression:
                        pred_q = F.softmax(logit_q, dim=1).argmax(dim=1)
                    
                for k in range(1, self.update_step):
                    # Run the i-th task and compute loss for k = 1
                    # Forward (bn_training)
                    logits = self.net(x_support[i], vars=task_parameter, bn_training=True)
                    
                    if not self._is_regression:
                        task_loss = F.cross_entropy(logits, y_support[i], reduction='sum')
                    else:
                        task_loss = F.mse_loss(logits, y_support[i])

                    # Gradient
                    grads = torch.autograd.grad(task_loss, task_parameter)

                    # Inner update
                    for j, (grad, param) in enumerate(zip(grads, task_parameter)):
                        new_weight = gradient_descent(grad, param, self.update_lr, is_hessian)
                        task_parameter[j] = new_weight

                    with torch.no_grad():
                        # Assessment
                        embedded_vector = self.net.get_embedded_vector(x_query[i], task_parameter, bn_training=True) #[n_way * k_shot_query, num_class]
                        embedded_vector = embedded_vector.view(n_way, k_shot_query, -1)

                # Append
                embedded_vector_list.append(embedded_vector)
                                
            # Reshape
            embedded_vector_stack = torch.stack(embedded_vector_list, dim=0) #[task_size, n_way, k_shot_query], type=torch.long
            
            return embedded_vector_stack


    


if __name__ == '__main__':    
    # Set Config
    config = list()
    config.append(('conv2d', [128, 3, 32, 32, 1, 0])) #[ch_out, ch_in, size, size, stride, padding]
    config.append(('flatten', []))
    config.append(('fc', [n_way, 128])) #[out_dim, in_dim]
    #config.append(('fc', [1, 3])) #[ch_out, ch_in, size, size, stride, padding]

    # Training 
    task_x_support = torch.Tensor(np.random.normal(size=(32, 5, 5, 3, 32, 32)))
    task_y_support = torch.Tensor(np.random.randint(low=0, high=n_way, size=(32, 5, 5))).long() # integer
    task_x_query = torch.Tensor(np.random.normal(size=(32, 5, 5, 3, 32, 32)))
    task_y_query = torch.Tensor(np.random.randint(low=0, high=1, size=(32, 5, 5))).long() # integer

    # Test
    x_support = torch.Tensor(np.random.normal(size=(1, 5, 5, 3, 32, 32)))
    y_support = torch.Tensor(np.random.randint(low=0, high=n_way, size=(1, 5, 5))).long() # integer
    x_query = torch.Tensor(np.random.normal(size=(1, 5, 5, 3, 32, 32)))
    y_query = torch.Tensor(np.random.randint(low=0, high=n_way, size=(1, 5, 5))).long()

    # Test forward
    Model = Meta(config)

    # Training
    meta_loss, meta_accuracy = Model.meta_loss(task_x_support, task_y_support, task_x_query, task_y_query)

    # Predict
    pred = Model(x_support, y_support, x_query)

    # Loss
    test_loss = torch.torch.eq(pred, y_support).sum().item() / (32 * 5 * 5)

    print(test_loss)








            

            

            

            









        

