import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stochastic_Conv2D(nn.Module):
    '''
        stochastic Conv_layer (bayes by backprop)
    '''


    def __init__(self, in_channel, out_channel, kernel_size, \
        stride=1, padding=0, dilation=1, bias=True, dist_params=None):

        super(Stochastic_Conv2D, self).__init__()

        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = (kernel_size, kernel_size)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = 1 # By default
        self._use_bias = bias

        if dist_params is None:
            dist_params = {
                'prior_mu' : 0,
                'prior_sigma' : 0.1,
                'posterior_mu_initial' : (0., 0.1),
                'posterior_rho_initial' : (-3, 0.1)
            }

        self._prior_mu = dist_params['prior_mu']
        self._prior_sigma = dist_params['prior_sigma']
        self._posterior_mu_initial = dist_params['posterior_mu_initial']
        self._posterior_rho_initial = dist_params['posterior_rho_initial']

        self._W_mu = nn.Parameter(torch.empty((out_channel, in_channel, *self._kernel_size)))
        self._W_rho = nn.Parameter(torch.empty((out_channel, in_channel, *self._kernel_size)))

        if self._use_bias:
            self._bias_mu = nn.Parameter(torch.empty((out_channel)))
            self._bias_rho = nn.Parameter(torch.empty((out_channel)))
        else:
            self._bias_mu, self._bias_rho = None, None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self._W_mu, *self._posterior_mu_initial)
        torch.nn.init.normal_(self._W_rho, *self._posterior_rho_initial)
        
        '''
        self._W_mu.data.normal_(*self._posterior_mu_initial)
        self._W_rho.data.normal_(*self._posterior_rho_initial)
        '''

        if self._use_bias:
            torch.nn.init.normal_(self._bias_mu, *self._posterior_mu_initial)
            torch.nn.init.normal_(self._bias_rho, *self._posterior_rho_initial)

    def forward(self, inputs, sample=True):
        if self.training or sample:
            self._W_sigma = torch.log1p(torch.exp(self._W_rho)) #torch.log1p : for small value (numerical stable)
            posterior_dist_w = torch.distributions.Normal(self._W_mu, self._W_sigma)
            weights = posterior_dist_w.rsample() #[out_channel, in_channel, kernel_size, kernel_size]

            if self._use_bias:
                self._bias_sigma = torch.log1p(torch.exp(self._bias_rho))
                posterior_dist_bias = torch.distributions.Normal(self._bias_mu, self._bias_sigma)
                biases = posterior_dist_bias.rsample()

            else:
                biases = None

        else:
            weights = self._W_mu
            biases = self._bias_mu if self._use_bias else None

        result = F.conv2d(inputs, weights, biases, self._stride, self._padding, self._dilation, self._groups)
        
        return result

    def kl_loss(self):
        posterior_dist_w = torch.distributions.Normal(self._W_mu, self._W_sigma)
        prior_dist_w = torch.distributions.Normal(self._prior_mu, self._prior_sigma)
        kld_weights = torch.distributions.kl.kl_divergence(posterior_dist_w, prior_dist_w).sum()

        if self._use_bias:
            posterior_dist_bias = torch.distributions.Normal(self._bias_mu, self._bias_sigma)
            prior_dist_bias = torch.distributions.Normal(self._prior_mu, self._prior_sigma) # broad casting 여부를 확인
            kld_biases = torch.distributions.kl.kl_divergence(posterior_dist_bias, prior_dist_bias).sum()

        kld = kld_weights + kld_biases

        return kld


class Stochastic_FC(nn.Module):
    '''
        stochastic FC_layer (bayes by backprop)
    '''

    def __init__(self, input_dim, output_dim, bias=True, dist_params=None):
        super(Stochastic_FC, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._use_bias = bias
        
        if dist_params is None:
            dist_params = {
                'prior_mu' : 0,
                'prior_sigma' : 0.1,
                'posterior_mu_initial' : (0., 0.1),
                'posterior_rho_initial' : (-3, 0.1)
            }

        self._prior_mu = dist_params['prior_mu']
        self._prior_sigma = dist_params['prior_sigma']
        self._posterior_mu_initial = dist_params['posterior_mu_initial']
        self._posterior_rho_initial = dist_params['posterior_rho_initial']

        # F.Linear operates like xA^T + b --> A (output_dim, input_dim)
        self._W_mu = nn.Parameter(torch.empty((output_dim, input_dim)))
        self._W_rho = nn.Parameter(torch.empty((output_dim, input_dim)))

        if self._use_bias:
            self._bias_mu = nn.Parameter(torch.empty((output_dim)))
            self._bias_rho = nn.Parameter(torch.empty((output_dim)))
        else:
            self._bias_mu, self._bias_rho = None, None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self._W_mu, *self._posterior_mu_initial)
        torch.nn.init.normal_(self._W_rho, *self._posterior_rho_initial)
        
        '''
        self._W_mu.data.normal_(*self._posterior_mu_initial)
        self._W_rho.data.normal_(*self._posterior_rho_initial)
        '''

        if self._use_bias:
            torch.nn.init.normal_(self._bias_mu, *self._posterior_mu_initial)
            torch.nn.init.normal_(self._bias_rho, *self._posterior_rho_initial)
            
    def forward(self, inputs, sample=True):
        if self.training or sample:
            self._W_sigma = torch.log1p(torch.exp(self._W_rho)) #torch.log1p : for small value (numerical stable)
            posterior_dist_w = torch.distributions.Normal(self._W_mu, self._W_sigma)
            weights = posterior_dist_w.rsample() #[input_dim, output_dim]

            if self._use_bias:
                self._bias_sigma = torch.log1p(torch.exp(self._bias_rho))
                posterior_dist_bias = torch.distributions.Normal(self._bias_mu, self._bias_sigma)
                biases = posterior_dist_bias.rsample()

            else:
                biases = None

        else:
            weights = self._W_mu
            biases = self._bias_mu if self._use_bias else None

        result = F.linear(inputs, weights, biases)
        

        return result

    def kl_loss(self):
        posterior_dist_w = torch.distributions.Normal(self._W_mu, self._W_sigma)
        prior_dist_w = torch.distributions.Normal(self._prior_mu, self._prior_sigma)
        kld_weights = torch.distributions.kl.kl_divergence(posterior_dist_w, prior_dist_w).sum()

        if self._use_bias:
            posterior_dist_bias = torch.distributions.Normal(self._bias_mu, self._bias_sigma)
            prior_dist_bias = torch.distributions.Normal(self._prior_mu, self._prior_sigma) # broad casting 여부를 확인
            kld_biases = torch.distributions.kl.kl_divergence(posterior_dist_bias, prior_dist_bias).sum()

        kld = kld_weights + kld_biases

        return kld







if __name__ == "__main__":
    BBB_Linear = Stochastic_FC(2, 1)

    inputs = torch.Tensor(np.random.normal(size=(3, 2)))
    outputs = torch.Tensor(np.random.normal(size=(3,)))

    pred = BBB_Linear(inputs)
    kld = BBB_Linear.kl_loss()

    mse = (outputs - pred).pow(2).sum()

    parameters = BBB_Linear.named_parameters()

    for i, params in enumerate(parameters):
        print(params[1])

    grad = torch.autograd.grad(mse, BBB_Linear.parameters())

    zips = zip(grad, BBB_Linear.parameters())

    for i, (grad, parameter) in enumerate(zip(grad, BBB_Linear.parameters())):
        print(grad)
        print(parameter)
        new_weights = parameter - 0.01 * grad
        
        print("new_weights {} : ".format(i), new_weights)

        

    fast_weights = list(map(lambda p : p[1] - 0.01 * p[0], zip(grad, BBB_Linear.parameters())))
    print("fast_weights : ", fast_weights)

    print(mse.item(), kld.item())





            




