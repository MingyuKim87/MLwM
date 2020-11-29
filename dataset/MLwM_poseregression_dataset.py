import os
import pwd

import numpy as np
import math
import pickle
import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

class meta_pose_regression_dataset(Dataset):
    '''
            Dataset for meta learning (few shots learning (support_set / query set))
                
                it consists of support_x, support_y, query_x, query_y
                    support_x : [num_way, num_point, channel, height, wdith]
                    support_y : [num_way, num_point]
                    query_x : [num_way, num_point, channel, height, wdith]
                    query_y : [num_way, num_point]

                self.data = [episode_length, (support_x, support_y, query_x, query_y)]

            - Cautions -
                when use the "intra_shuffle" type, you have to run self.reset_episode prior to sample data

            type : 
                1. non_mutual_exclusive : all individual tasks have same classes and labels during training and testing
                2. intra_shuffle : all individual tasks have same classes during training and testing, 
                    but the labels can be switched. (sometimes [0, 1, 2, 3,4] and [2,1,3,4,5] etc...)
                3. Inter_shuffle : we completely do random selection of classes for making tasks. 
            
            1) set the parameters of meta learning dataset 
                : __init__
            2) load the miniimagenet dataset which consists of all images and target values
                : self.miniimagenet_dataset
            3) make a episodes (select tasks). It means a epochs
                : store at self.data called by self._generate_episode
            4) make a each task (select the number of points)
                : in self._generate_eposide, we sample random numbers for task     
    '''

    def __init__(self, n_way, k_shot_support, k_shot_query, root_path, \
        x_transform=None, y_transform=None, mode=None, types='non_mutual_exclusive'):

        '''
            Args
                mode : training set or validation set
                types : data types
        '''

        # Data Import
        if x_transform is None:
            # On Numpy
            self._x_transform = transforms.Compose([lambda x: x / 255.0])
        else:
            self._x_transform = x_transform

        if y_transform is None:    
            self._y_transform = transforms.Compose([lambda y: y * 10.0])
        else:
            self._y_transform = y_transform

        # Define input and output
        self.inputs, self.outputs = self._get_input_and_target(root_path, mode=mode)

        # Define
        self.num_classes = self._get_num_classes()
        self.min_points = self._get_num_datapoint()

        # Define pose_regression_dataset
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.types = types

        # Exceptional Treatment
        assert self.k_shot_query + self.k_shot_support <= self.min_points
        
        # Define episode length
        self.episode_length = self._get_episode_length(self.num_classes, self.n_way)

        # Data
        self.data = self._generate_episode(self.inputs)

    
    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        return self.data[index]

    def _get_num_classes(self):
        num_classes_x = self.inputs.shape[0]
        num_classes_y = self.outputs.shape[0]

        assert num_classes_x == num_classes_y

        return num_classes_x

    def _get_num_datapoint(self):
        num_datapoints_x = self.inputs.shape[1]
        num_datapoints_y = self.outputs.shape[1]

        assert num_datapoints_x == num_datapoints_y

        return num_datapoints_x
    
    def _generate_episode(self, inputs):
        '''
            Generate a meta-learning episode    

            Args : inputs [num_classes, num_points, channel, height, width]

            Return : a episode
                episode is data set for a epoch (round) [num_task, 4(support_x, support_y, query_x, query_y)]
                    support_x : [n_way, k_shot, channel, height, width]
                    support_y : [n_way, k_shot]
                    query_x : [n_way, k_shot, channel, height, width]
                    query_y : [n_way, k_shot]

        '''
        data = []
        
        # Permutation of num_points
        perm_num_points = np.random.permutation(self.min_points)
        self.inputs = self.inputs[:, perm_num_points]

        # Random choice of k_shot_support and k_shot_query
        num_point_index = np.random.choice(self.min_points, \
            (self.k_shot_support + self.k_shot_query), False)

        # Generate a episode list
            # Pose regression have only non-mutual-exclusive types
        episode_task_list = self._get_episode_task_index(\
            self.num_classes, self.n_way, self.types) #[episode_length, n_way]

        for task_list in episode_task_list:
            # Generate task index : [num_way, k_shot_support + k_shot_query]
            task_list_index = np.tile(task_list[:, None], \
                (1, self.k_shot_support + self.k_shot_query))

            # Generate num_point index : [num_way, k_shot_support + k_shot_query]
            task_num_point_index = np.tile(num_point_index, (task_list.shape[0],1))

            # Sample from inputs
            task_inputs = self.inputs[task_list_index, task_num_point_index]

            # Sample from outputs
            
            
            task_outputs = self.outputs[task_list_index, task_num_point_index]

            if self.types=="inter_shuffle":
                shape = task_outputs.shape
                noise = np.random.normal(size=shape)
                task_outputs = task_outputs + noise

            # support / query set
            task_inputs_support = task_inputs[:, :self.k_shot_support] #[n_way, k_shot, channel, width, height]
            task_inputs_query = task_inputs[:, self.k_shot_support:] #[n_way, k_shot, channel, width, height]
            task_target_support = task_outputs[:, :self.k_shot_support] #[n_way, k_shot, 1]
            task_target_query = task_outputs[:, self.k_shot_support:] #[n_way, k_shot, 1]

            '''
            # Transform target dims [n_way, k_shots, 1] --> [n_way, k_shot]
            task_target_support = np.squeeze(task_target_support, -1)
            task_target_query = np.squeeze(task_target_query, -1)
            '''

            data.append((task_inputs_support, task_target_support, \
                task_inputs_query, task_target_query))

        return data #[episode_length, 4]

    def _get_input_and_target(self, root_path, mode):
        '''
            Transform all items into input and output 
                if dataset are composed of torch.Tensor, returns should be also torch.Tensor
                but, if dataset are composed of np.array, returns should be also np.array

            Args:
                dir_path : the path of data directory
                data_path_list : the filename of train or test set. 
                mode : validation set or training set
            
            Return : 
                # torch.Tensor or np.array
                inputs : input images [num_classes, num_points, channel, width, height]
                ouput : output classes [num_classes]
        '''
        # Define filenames        
        if mode=='val':
            dataset_file_path = 'val_data_2.pkl'
        else:
            dataset_file_path = 'train_data_2.pkl'

        
        # Load data
        x, y = pickle.load(\
            open(os.path.join(root_path, dataset_file_path), 'rb'))

        x, y = np.array(x), np.array(y)

        # Drop the roll, pitch and maintain the yaw dimension
            # The reason why The last comonent "none" is that 
                # we keep the y dimension #[num_classes, ]
        y = y[:, :, -1, None]

        # Add channel dimension
        x = x[:, :, None, :, :]


        inputs = self._x_transform(x)
        outputs = self._y_transform(y)

        return inputs, outputs

    def _get_episode_task_index(self, num_classes, n_way, type):
        '''
            make a episode (epoch) which consists of set of num_classes
                "non_mutual_exclusive" or "intra_shuffle" 
                    : all tasks consists of the fixed classes (increasing order)
                "inter_shuffle"
                    : random choose the classes

            Args : 
                num_classes : total number of classes (Omniglot)
                n_way : the number of ways

            Returns : 
                index_list : A sequence of task have a list of class set. [episode_length, n_way]
        '''
        
        # Exceptional Treatment
        assert type == "non_mutual_exclusive" or \
            type == "intra_shuffle" or \
                type == "inter_shuffle"

        index_list = []

        for i in range(0, num_classes, n_way):
            if i + n_way > num_classes:
                continue
            else:
                index_list.append(np.array(range(i,i+n_way)))

        return np.array(index_list)

    def _get_episode_length(self, num_classes, n_way):
        return math.floor(num_classes/n_way)

    def reset_episode(self):
        self.data = self._generate_episode(self.inputs)
        return

if __name__ == "__main__":
    DEBUG = True
    
    pose_regression_filepath = '/home/mgyukim/Data/rotate_resize/Dataset'
    dataset = meta_pose_regression_dataset(5, 5, 5, pose_regression_filepath, types='non_mutual_exclusive')
    dataset.reset_episode()

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)
        break

    sample_image = support_x[0, 0, 0]
    sample_image = sample_image * 255

    sample_image_np = sample_image.numpy()
    sample_image_np = np.transpose(sample_image_np, (1,2,0))
    
    img = Image.fromarray(sample_image_np[:, :, -1])
    img = img.convert("L")
    img.save("./test_image.png")

    print(support_y[0,0,0])

    sample_image = query_x[0, 0, 0]
    sample_image = sample_image * 255

    sample_image_np = sample_image.numpy()
    sample_image_np = np.transpose(sample_image_np, (1,2,0))
    
    img = Image.fromarray(sample_image_np[:, :, -1])
    img = img.convert("L")
    img.save("./test_image_query.png")

    print(query_y[0,0,0])

