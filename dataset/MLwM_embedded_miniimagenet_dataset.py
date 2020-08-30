import os
import pwd

import numpy as np
import math
import pickle
import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class meta_embedded_miniimagenet_dataset(Dataset):
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
        mode=None, types='non_mutual_exclusive'):

        '''
            Args
                mode : training set or validation set or test set
                types : data types
        '''

        # Define input and output
        self.inputs, self.outputs = self._get_input_and_target(root_path, mode=mode)

        # Define
        self.num_classes = self._get_num_classes()
        self.min_num_points = self._get_num_datapoint()

        # Define Omniglot_dataset
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.types = types

        # Exceptional Treatment
        assert self.k_shot_query + self.k_shot_support <= self.min_num_points
        
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
        
        if self.types == "inter_shuffle":
            perm = np.random.permutation(self.num_classes)
            self.inputs = self.inputs[perm] # random permutation according to class index

        # Permutation of num_points
        perm_num_points = np.random.permutation(self.min_num_points)
        self.inputs = self.inputs[:, perm_num_points]

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
        if mode=='test':
            dataset_file_path = 'test_embeddings.pkl'
        if mode=='val':
            dataset_file_path = 'val_embeddings.pkl'
        else:
            dataset_file_path = 'train_embeddings.pkl'
        
        # Load data
        data = pickle.load(\
            open(os.path.join(root_path, dataset_file_path), 'rb'), encoding='latin1')

        # Construct data
        img_by_class, embed_by_name, class_list = self.construct_embedding_data(data)

        # create_embedding_list
        inputs, outputs = self.create_embedding_list(img_by_class, embed_by_name, class_list)

        return inputs, outputs

    def create_embedding_list(self, img_by_class, embed_by_name, class_list, is_torch=False):
        container = {}
        num_class = 0

        inputs = []
        outputs = []
        

        for key, value in embed_by_name.items():
            img = value
            label, img_name = key.split('_')

            if label in container.keys():
                container[label].append(img)
            else:
                container[label] = [img]

        # For input and output
        for index, (label, img_list) in enumerate(container.items()):
            # img_list : [20, ], each compoenent is torch.Tensor [channel, height, width]
            
            if is_torch:
                # img_list = tensor, [20, channle, height, width]
                img_list = torch.stack(img_list)
            
            inputs.append(img_list)
            outputs.append(index)

        if is_torch:
            # List -> torch.Tensor
            inputs = torch.stack(inputs)
            outputs = torch.LongTensor(outputs)
        else :
            # List -> np.array
            inputs = np.array(inputs)
            outputs = np.array(outputs)

        return inputs, outputs

    def construct_embedding_data(self, data):
        image_by_class = {}
        embed_by_name = {}
        class_list = set()

        keys = data["keys"]

        for i, k in enumerate(keys):
            _, class_name, img_name = k.split('-')
            
            if (class_name not in image_by_class):
                image_by_class[class_name] = []

            image_by_class[class_name].append(img_name)
            embed_by_name[img_name] = data["embeddings"][i]

            # Construct class list
            class_list.add(class_name)

        class_list = list(class_list)

        return image_by_class, embed_by_name, class_list
        
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
    
    embedding_miniimagenet_path = '/home/mgyukim/Data/embeddings/miniImageNet/center'
    dataset = meta_embedded_miniimagenet_dataset(5, 5, 5, embedding_miniimagenet_path, mode='train')
    dataset.reset_episode()

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)
        break

    print(support_y)

    print(query_y)