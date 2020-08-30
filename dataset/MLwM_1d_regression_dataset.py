import os
import pwd

import math
import numpy as np
import GPy

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class meta_1d_regression_GP_dataset(Dataset):
    '''
            Dataset for meta learning (few shots learning (support_set / query set))
                
                it consists of support_x, support_y, query_x, query_y
                    support_x : [num_way, num_point, channel, height, wdith]
                    support_y : [num_way, num_point]
                    query_x : [num_way, num_point, channel, height, wdith]
                    query_y : [num_way, num_point]

                self.data = [episode_length, (support_x, support_y, query_x, query_y)]

            1) set the parameters of meta learning dataset 
                : __init__
            2) make a gp function for generating points
                : self._generate_functions()
            3) make a episodes (select tasks). It means a epochs
                : store at self.data called by self._generate_episode
            4) make a each task (select the number of points)
                : in self._generate_eposide, we sample random numbers for task     
    '''

    def __init__(self, k_shot_support, k_shot_query, l1_scale=.4, simga_scale=1., mode='RBF'):
        '''
            if train : k_shot_query = 5 or 10
            if valid : k_shot_query = 100 or above
        '''

        # Define few shots learning parameters
        self.n_way = 1
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.num_points = self.k_shot_support + self.k_shot_query
        self.total_tasks = 100

        # Define gp parameters
        self.mode = mode # two kernel GP or RBF GP
        self._y_dim = 1
        self._x_dim = 1
        self._l1_scale = l1_scale
        self._sigma_scale = simga_scale
        
        # Initial inputs and outputs
        self.inputs, self.outputs = self._get_input_and_target()

        # Data
        self.data = self._generate_episode()

    def __len__(self):
        return self.total_tasks

    def __getitem__(self, index):
        return self.data[index]

    def _rbf_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        '''
            RBF kernel to generate GPs using 'Numpy'
        
            Args:
                xdata : [num_total_points, x_size(dimension)]
                l1 : [y_size(dimension), x_size(dimension)], the scale parameters of RBF kernel
                sigma_f : float, [y_size(dimension)], GP's std
                sigma_noise : float, noise for GP's std

            Returns:
                The kernel [batch_size, y_size, num_total_points, num_total_points]
        '''

        num_total_points = xdata.shape[0]

        # Expand
        xdata1 = xdata[None, :, :]
        xdata2 = xdata[:, None, :]
        diff = xdata1 - xdata2 #diff : [num_total_points, num_total_points, x_size(dimension)] <- "broadcasting"

        # L2 Norm
        l2_norm = np.square(diff[None, :, :, :]) #diff : [1, num_total_points, num_total_points, x_size(dimension)] 
        scaling_rbf = l1[:, None, None, :] #l1 : [y_size(dimension), 1, 1, x_size(dimension)] 

        l2_norm = l2_norm / scaling_rbf # l2_norm with scalaing factor [1, num_total_points, num_total_points, x_size(dimension)] 
        l2_norm = np.sum(l2_norm, axis=-1) #[1, num_total_points, num_total_points] 

        # Kernel
        constant_kernel = np.square(sigma_f)[:, None, None] #[y_size, 1, 1] 
        kernel = constant_kernel * np.exp(-.5 * l2_norm) #[1, num_total_points, num_total_points] 

        #kernel + bias
        kernel += (sigma_noise ** 2) * np.identity(num_total_points) #[1, num_total_points, num_total_points] 

        return kernel

    def _get_input_and_target(self):
        '''
            Generate a meta-learning episode    

            Args : self.num_point, self.x_dim, self.y_dim, self._l1_scale, self._sigma_scale

            Return : GP datasets
                x = [num_points, x_dim]
                y = [num_points, y_dim]
        '''

        num_total_points = self.num_points
        range_x_value = np.linspace(-4., 4., num=num_total_points) #[the number of total points, ] 
        # When testing phase, all batch have same elements (batch(0))
        x_values = np.expand_dims(range_x_value, axis=-1) # [num_total_points, 1] # this function only work well "x_size = 1"
    
        # Data generation by GPs
            # Set Kernel Paraemters : homogeneous covariance matrix
        k1 = GPy.kern.RBF(input_dim=self._x_dim, variance=self._sigma_scale, lengthscale=self._l1_scale)
        k2 = GPy.kern.src.periodic.PeriodicExponential(period=np.pi, n_freq=30)

        # Covariance matrix
        if self.mode == "RBF":
            C = k1.K(x_values, x_values)
        elif self.mode == "two_kernel":
            C = k1.K(x_values, x_values) + k2.K(x_values, x_values)
        else :
            assert NotImplementedError

        # Sampling
        y = np.random.multivariate_normal(np.zeros(self.num_points), C)[:, None] #[num_points, y_dim]
        y_values = (y - y.mean())

        return x_values.astype(np.float32), y_values.astype(np.float32)

    def _generate_episode(self):
        '''
            Generate a meta-learning episode    

            Args : inputs [num_total_points, 1]

            Return : a episode
                episode is data set for a epoch (round) [num_task, 4(support_x, support_y, query_x, query_y)]
                    support_x : [1, k_shot, 1]
                    support_y : [1, k_shot, 1]
                    query_x : [1, k_shot, 1]
                    query_y : [1, k_shot, 1]
        '''

        data = []

        for _ in range(self.total_tasks):

            # generate gp dataset
            self.inputs, self.outputs = self._get_input_and_target()

            # Random index
            random_index = np.random.permutation(self.num_points)
            support_index = random_index[:self.k_shot_support]
            query_index = random_index[self.k_shot_support:]

            # Split data
            support_x_temp = self.inputs[support_index]
            support_y_temp = self.outputs[support_index]
            query_x_temp = self.inputs[query_index]
            query_y_temp = self.outputs[query_index]

            # add n_way dim
            support_x = support_x_temp[None, :, :] #[n_way(1), k_shots, x_dim]
            support_y = support_y_temp[None, :, :] #[n_way(1), k_shots, y_dim]
            query_x = query_x_temp[None, :, :] #[n_way(1), k_shots, x_dim]
            query_y = query_y_temp[None, :, :] #[n_way(1), k_shots, y_dim]

            data.append((support_x, support_y, query_x, query_y))

        return data

    def reset_episode(self):
        self.data = self._generate_episode()
        return


if __name__ == "__main__":
    DEBUG = True
    
    train_dataset = meta_1d_regression_GP_dataset(5, 5)
    valid_dataset = meta_1d_regression_GP_dataset(5, 100)
    train_dataset.reset_episode()
    valid_dataset.reset_episode()

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)

        break

    for i, data in enumerate(valid_dataloader):
        support_x, support_y, query_x, query_y = data
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)

        break

    x = query_x[0]
    y = query_y[0]

    x = x.view(-1)
    y = y.view(-1)

    plt.scatter(x.numpy(), y.numpy())
    plt.savefig("./test.png")








