import os
import pwd

import numpy as np
import math
import pickle
import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


import functools


from PIL import Image
import matplotlib.pyplot as plt

class miniImagenet_dataset(Dataset):
    '''
        put mini-imagenet files as :
            root :
                |- images/*.jpg includes all imgeas
                |- train.csv
                |- test.csv
                |- val.csv
    '''
    def __init__(self, root_dir, mode=None, transform=None):

        # Set parameter
        self._root_dir = root_dir

        # Default transform
        if transform == None:
            self._transform = transforms.Compose([lambda x : Image.open(x).convert('RGB'),\
                lambda x : np.array(x)])
        else:
            self._transform = transform
        
        # Set directory names
        self._csv_directory = 'csvfiles'
            
        if mode == "test":
            self._filename = 'test.csv'
        elif mode == "val":
            self._filename = 'val.csv'
        else:
            self._filename = 'train.csv'

        # Set file_path
        self._file_list = self._loadCSV(os.path.join(self._root_dir, 'csvfiles', self._filename))
        self._target_classes, self._min_data_points = self._indexing_classes(self._file_list)

    def __len__(self):
        total_dataset_length = self._file_list.shape[0]
        return total_dataset_length

    def __getitem__(self, index):
        '''
            input : 
                self._file_liast = [(file_path, class_name)]

            return : 
                img, target
        '''

        file_path = os.path.join(self._root_dir, 'images', self._file_list[index][0])

        # Transform a file path to img
        img = self._transform(file_path)
        
        # Target information
        target = self._target_classes[self._file_list[index][1]]

        return img, target

    def _loadCSV(self, filepath):
        file_data = np.genfromtxt(filepath, dtype=None, delimiter=',', encoding='UTF-8')[1:]
        return file_data

    def _indexing_classes(self, filelist):
        # Classes
        target_classes = {}

        # class_numbers 
        class_datapoint = {}


        for i, item in enumerate(filelist):
            '''
                i : index
                item : (file_name, class_name) i.e) n0153282900000005.jpg,n01532829
            '''
            # dict type으로 container를 설정할 경우, key값으로 not in operator를 사용할 수 있다. 
            if item[1] not in target_classes :
                target_classes[item[1]] = len(target_classes)
                class_datapoint[item[1]] = 1
            else:
                class_datapoint[item[1]] += 1

        # minimum_values in class_datapoints
        min_data_points = min(class_datapoint.values())
                
        return target_classes, min_data_points

    def get_num_classes(self):
        return len(self._target_classes)

    def get_num_min_datapoint(self):
        return self._min_data_points

class meta_miniImagenet_dataset(Dataset):
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

    def __init__(self, n_way, k_shot_support, k_shot_query, root_path, img_size, transform=None, mode=None, types='non_mutual_exclusive', perm_index_list=None):
        # Set Image Size
        self._img_size = img_size
        
        # Data Import
        if transform is None:
            '''
            # On numpy
            self._transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),\
                                        lambda x : x.resize((img_size, img_size)),\
                                        lambda x : np.array(x),\
                                        ])
            '''

            # On torch
            self._transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((img_size, img_size)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self._transform = transform
        
        # Set miniImageNet dataset 
        self._miniimagenet_dataset = miniImagenet_dataset(root_path, mode=mode, transform=self._transform)

        # Define input and output
        self.inputs, self.outputs = self._get_input_and_target(self._miniimagenet_dataset)

        # Define
        self.num_classes = self._miniimagenet_dataset.get_num_classes()
        self.min_num_points = self._miniimagenet_dataset.get_num_min_datapoint()

        # Define Omniglot_dataset
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.types = types

        # Exceptional Treatment
        assert self.k_shot_query + self.k_shot_support <= self.min_num_points
        
        # Define episode length
        self.episode_length = self._get_episode_length(self.num_classes, self.n_way)

        # Permutation list
        self.perm_index_list = perm_index_list

        # Data
        self.data = self._generate_episode(self.inputs)

    
    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        return self.data[index]

    def set_perm_index_list(self, perm_list):
        self.perm_index_list = perm_list
    
    def _generate_episode(self, inputs, **kwargs):
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
        
        if self.types == "inter_shuffle":
            perm = np.random.permutation(self.num_classes)
            self.inputs = self.inputs[perm] # random permutation according to class index

        if self.perm_index_list is None:
            # Permutation of num_points
            perm_num_points = np.random.permutation(self.min_num_points)
            self.inputs = self.inputs[:, perm_num_points]
        else:
            self.inputs = self.inputs[:, self.perm_index_list]

        # Random choice of k_shot_support and k_shot_query
        num_point_index = np.random.choice(self.min_num_points, \
            (self.k_shot_support + self.k_shot_query), False)

        # Generate a episode list
        episode_task_list = self._get_episode_task_index(\
            self.num_classes, self.n_way, self.types) #[episode_length, n_way]

        for task_list in episode_task_list:
            labels = np.tile(np.array(range(self.n_way))[:, None], \
                (1, self.k_shot_support + self.k_shot_query))
            
            # Generate task index : [num_way, k_shot_support + k_shot_query]
            task_list_index = np.tile(task_list[:, None], \
                (1, self.k_shot_support + self.k_shot_query))

            # Generate num_point index : [num_way, k_shot_support + k_shot_query]
            task_num_point_index = np.tile(num_point_index, (task_list.shape[0],1))

            # Sample from inputs
            task_inputs = self.inputs[task_list_index, task_num_point_index]

            # support / query set
            task_inputs_support = task_inputs[:, :self.k_shot_support] #[n_way, k_shot, channel, width, height]
            task_inputs_query = task_inputs[:, self.k_shot_support:] #[n_way, k_shot, channel, width, height]
            task_target_support = labels[:, :self.k_shot_support] #[n_way, k_shot]
            task_target_query = labels[:, self.k_shot_support:] #[n_way, k_shot]

            data.append((task_inputs_support, task_target_support, \
                task_inputs_query, task_target_query))

        return data #[episode_length, 4]

    def _get_input_and_target(self, dataset):
        '''
            Transform all items into input and output 
                if dataset are composed of torch.Tensor, returns should be also torch.Tensor
                but, if dataset are composed of np.array, returns should be also np.array

            Args:
                dataset : minimimagenet_dataset which have a all items
            Return : 
                # torch.Tensor or np.array
                inputs : input images [num_classes, num_points, channel, width, height]
                ouput : output classes [num_classes]
        '''
        # Initialize
        container = {}
        inputs = []
        outputs = []

        # Check out the dataset comes from torch.Tensor
        img, label = dataset[0]
        is_torch = torch.is_tensor(img)

        # For containeres
        for i in range(len(dataset)):
            # Get items
            img, label = dataset[i]

            if label in container.keys():
                container[label].append(img)
            else:
                container[label] = [img]


        # For input and output
        for label, img_list in container.items():
            # img_list : [20, ], each compoenent is torch.Tensor [channel, height, width]
            
            if is_torch:
                # img_list = tensor, [20, channle, height, width]
                img_list = torch.stack(img_list)
            
            inputs.append(img_list)
            outputs.append(label)

        
        if is_torch:
            # List -> torch.Tensor
            inputs = torch.stack(inputs)
            outputs = torch.LongTensor(outputs)
        else :
            # List -> np.array
            inputs = np.array(inputs)
            outputs = np.array(outputs)

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
            
            if type == "non_mutual_exclusive" or type == "intra_shuffle":
                index_list.append(np.array(range(i,i+n_way)))

            elif type == "inter_shuffle":
                random_index = np.random.choice(range(num_classes), n_way, replace=False).tolist()
                index_list.append(random_index)

            else:
                assert NotImplementedError

        return np.array(index_list)

    def _get_episode_length(self, num_classes, n_way):
        return math.floor(num_classes/n_way)

    def reset_episode(self):
        self.data = self._generate_episode(self.inputs)
        return


if __name__ == "__main__":
    DEBUG = True
    
    miniImagenet_filepath = "/home/mgyukim/Data/miniimagenet"
    dataset = meta_miniImagenet_dataset(5, 5, 5, miniImagenet_filepath, 28, types='inter_shuffle')
    dataset.reset_episode()

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data
        print(support_x.shape)



    

    
        

    



