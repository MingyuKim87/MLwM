import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class Learner(nn.Module):
    '''
        Base layer (FC, Conv, Convt2d, BatchNorm)
    '''

    def __init__(self, config, initializer=None):
        '''
            config : select a layer type (Conv, FC, Convt2d, BatchNorm)
            imgc : channel of images (Optional)
            imgsz : width or height of images (Optional)
        '''

        super(Learner, self).__init__()

        # This variable contains all tensors should be optimized. 
        self.config = config

        # Initializer
        if initializer is None:
            self.initializer = torch.nn.init.kaiming_normal_
        else:
            self.initializer = initializer

        # All variables are stored in self.vars
        self.vars = nn.ParameterList()
        # var_bn has constant variables which don't require to be optimized. 
        self.vars_bn = nn.ParameterList()

        # self.config is a "dict" 
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernel_size, kernel_size]
                    # param[:4] is list 
                w = nn.Parameter(torch.empty(*param[:4]))
                b = nn.Parameter(torch.empty(param[0]))

                # Initialization
                self.initializer(w)
                torch.nn.init.zeros_(b)

                # append
                self.vars.append(w)
                self.vars.append(b)

            elif name == 'convt2d':
                # [ch_int, ch_out, kernel_size, kernel_size, stride, padding]
                    # gain=1 according to cbfin's implementation
                w = nn.Parameter(torch.empty(*param[:4]))
                b = nn.Parameter(torch.empty(param[1]))

                # Initialization
                self.initializer(w)
                torch.nn.init.zeros_(b)

                # append
                self.vars.append(w)
                self.vars.append(b)

            elif name == "fc":
                # [input_dim, output_dim]
                w = nn.Parameter(torch.empty(*param))
                b = nn.Parameter(torch.empty(param[0]))

                # Initialization
                self.initializer(w)
                torch.nn.init.zeros_(b)

                # Append
                self.vars.append(w)
                self.vars.append(b)

            elif name == "bn":
                # [output_dim]
                w = nn.Parameter(torch.empty(param[0]))
                b = nn.Parameter(torch.empty(param[0]))

                # initialization
                torch.nn.init.ones_(w)
                torch.nn.init.zeros_(b)

                # Append
                self.vars.append(w)
                self.vars.append(b)

                # Must set requires_grad = False
                running_mean = nn.Parameter(torch.empty(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.empty(param[0]), requires_grad=False)

                # initialization
                torch.nn.init.zeros_(running_mean)
                torch.nn.init.ones_(running_var)

                # Extend  : "Appends parameters from a Python iterable to the end of the list."
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', \
                'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        '''
            Set variable name of each layer
        '''
        
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = "conv2d: (ch_in:{}, ch_out:{}, k:{} x {}, stride:{}, padding:{})".format(\
                    param[1], param[0], param[2], param[3], param[4], param[5])
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:{}, ch_out:{}, k:{} x {}, stride:{}, padding:{})'.format(\
                    param[0], param[1], param[2], param[3], param[4], param[5])
                info += tmp + '\n'

            elif name == 'fc':    
                tmp = 'linear:(in: {}, out: {})'.format(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope: {})'.format(param[0])
                info += tmp + '\n'

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k: {}, stride: {}, padding: {})'.format(\
                    param[0], param[1], param[2])
                info += tmp + '\n'

            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k: {}, stride: {}, padding: {})'.format(\
                    param[0], param[1], param[2])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample',\
                 'reshape', 'sigmoid', 'use_logits', 'bn']:
                 tmp = name + ':' + str(tuple(param))
                 info += tmp + '\n'

            else:
                raise NotImplementedError


    def forward(self, x, vars=None, bn_training=True, DEBUG=False):
        """

        This function can be called by finetunning (task specific parameters), 
        however, in finetunning, we don't wish to update running_mean/running_var.
        Thought weights/bias of bn is updated, it has been separated by task specific parameters.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via task spefici parameters.

        :param x: [b, 1, 28, 28]

        :param vars: model 

        :param bn_training: set False to not update

        :return: x, 

        """

        if vars is None:
            vars = self.vars

        idx = 0 
        bn_idx = 0

        hidden = x

        for name, param in self.config:
            
            # Weights
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]

                # remember to keep synchrozied of forward_encoder and forward_decoder
                hidden = F.conv2d(hidden, w, b, stride=param[4], padding=param[5])
                idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'convt2d':
                w, b = vars[idx], vars[idx+1]
                # remember to keep synchrozied of forward_encoder and forward_decoder
                hidden = F.conv_transpose2d(hidden, w, b, stride=param[4], padding=param[5])
                idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'fc':
                w, b = vars[idx], vars[idx+1]
                # remember to keep synchrozied of forward_encoder and forward_decoder
                hidden = F.linear(hidden, w, b)
                idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'bn':
                w, b = vars[idx], vars[idx+1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]

                hidden = F.batch_norm(hidden, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            # Activations
            elif name == 'flatten':
                if DEBUG == True:
                    print(name, param, "Before flatten shape: ", hidden.shape)
                
                hidden = hidden.view(hidden.size(0), -1) # synchronize the number of data point

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'reshape':
                hidden = hidden.view(hidden.size(0), *param)

            elif name == 'relu':
                hidden = F.relu(hidden, inplace=param[0])

            elif name == 'reakyrelu':
                hidden = F.leaky_relu(hidden, negative_slope=param[0], inplace=param[1])

            elif name == 'tanh':
                hidden = F.tanh(hidden)

            elif name == 'sigmoid':
                hidden = F.sigmoid(hidden)

            elif name == 'upsample':
                hidden = F.upsample_nearest(hidden, scale_factor=param[0])

            elif name == 'max_pool2d':
                hidden = F.max_pool2d(hidden, param[0], param[1], param[2])

            elif name == 'avg_pool2d':
                hidden = F.avg_pool2d(hidden, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return hidden


    def zero_grad(self, vars=None):
        '''
            initialize 'grad' member variables of each parameters
            
            param : vars (nn.parameterlist)

            returns
        '''

        # change "require_grad" from True to False
            # if grad is not none, all grad member variables set "zero"

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()

            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        '''
            override this function since initial parameters will return with a generator.
        '''
        return self.vars

    def get_embedded_vector(self, x, vars=None, bn_training=True, DEBUG=False):
        '''
            get_embedded_vector(self)

            return : embedded vectors (n_way * k_shot, 800)
        '''

        if vars is None:
            vars = self.vars

        idx = 0 
        bn_idx = 0

        hidden = x

        for name, param in self.config:    
            # Weights
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]

                # remember to keep synchrozied of forward_encoder and forward_decoder
                hidden = F.conv2d(hidden, w, b, stride=param[4], padding=param[5])
                idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'convt2d':
                w, b = vars[idx], vars[idx+1]
                # remember to keep synchrozied of forward_encoder and forward_decoder
                hidden = F.conv_transpose2d(hidden, w, b, stride=param[4], padding=param[5])
                idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'fc':
                break

            elif name == 'bn':
                w, b = vars[idx], vars[idx+1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]

                hidden = F.batch_norm(hidden, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            # Activations
            elif name == 'flatten':
                if DEBUG == True:
                    print(name, param, "Before flatten shape: ", hidden.shape)
                
                hidden = hidden.view(hidden.size(0), -1) # synchronize the number of data point

                if DEBUG == True:
                    print(name, param, "shape: ", hidden.shape)

            elif name == 'reshape':
                hidden = hidden.view(hidden.size(0), *param)

            elif name == 'relu':
                hidden = F.relu(hidden, inplace=param[0])

            elif name == 'reakyrelu':
                hidden = F.leaky_relu(hidden, negative_slope=param[0], inplace=param[1])

            elif name == 'tanh':
                hidden = F.tanh(hidden)

            elif name == 'sigmoid':
                hidden = F.sigmoid(hidden)

            elif name == 'upsample':
                hidden = F.upsample_nearest(hidden, scale_factor=param[0])

            elif name == 'max_pool2d':
                hidden = F.max_pool2d(hidden, param[0], param[1], param[2])

            elif name == 'avg_pool2d':
                hidden = F.avg_pool2d(hidden, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        return hidden





if __name__ == '__main__':
    # Set Config
    config = list()
    config.append(('conv2d', [128, 3, 32, 32, 1, 0])) #[ch_out, ch_in, size, size, stride, padding]
    config.append(('flatten', []))
    config.append(('fc', [1, 128]))
    #config.append(('fc', [1, 3])) #[ch_out, ch_in, size, size, stride, padding]

    # Input
    inputs = torch.Tensor(np.random.normal(size=(64, 3, 32, 32)))
    outputs = torch.Tensor(np.random.randint(0, 1, size=(64,)))

    Layer_1 = Learner(config)

    # Predict
    pred = Layer_1(inputs, DEBUG=True)

    # Loss
    loss = torch.nn.functional.mse_loss(pred, outputs)

    # Parameters
    parameters = Layer_1.named_parameters()
    parameters_1 = Layer_1.parameters()

    for i, params in enumerate(parameters_1):
        print(i)

    for i, params in enumerate(parameters):
        print(i)

    # Grad
    grad = torch.autograd.grad(loss, Layer_1.parameters())


    fast_weights_1 = []
    for i, (grad, parameter) in enumerate(zip(grad, Layer_1.parameters())):
        print(grad)
        print(parameter)
        new_weights = parameter - 0.01 * grad
        
        print("new_weights {} : ".format(i), new_weights)

        fast_weights_1.append(new_weights)

    # Weight list
    fast_weights = list(map(gradient_descent, zip(grad, parameters_1))) # zip으로 묶을 수가 없다. parameter_1 이라는 객체가 일반 list가 아니기 때문이다. 

    

    # print
    print("fast_weights : ", fast_weights)
    print("fast_weights_1 :", fast_weights_1)

    # print
    print("fast_weights (shape) : ", len(fast_weights))
    print("fast_weights_1 (shape) :", len(fast_weights_1))

    





            







    
    

    


                
                




            


            





                





        

