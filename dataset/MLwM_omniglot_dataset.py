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

def find_classes(root_dir):
    '''
        List up all files 
        
        Arg:
            root_dir : root dir (string)
                
        Returns :
            retour : a list of all files ([file_path, directory_name, root_dir])
    '''

    retour = []
    
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr-2] + "/" + r[lr-1], root))

    #print("="*5, "Found {} items".format(len(retour)), "="*5)

    return retour

def index_classes(items):
    '''
        Transform all items into [n_class, n_point, ]
        and generate a target_class

        Arg:
            items : a list of all items of Omniglot 
                [(file_path, directory_name, root_dir)]
        Returns :
            x : a list of file_paths [n_class, n_points, ]
                each component is (file_path, directory_name, root_dir)
            target_classes : a dict of classes
                each component is (directory_name, class)
    '''
    
    # Classes
    target_classes = {}

    # Features
    x = []

    # Total Number
    total_length = len(items)

    for i, item in enumerate(items):
        # dict type으로 container를 설정할 경우, key값으로 not in operator를 사용할 수 있다. 
        if item[1] not in target_classes :
            if not i == 0:
                x.append(temp) # 이전 클래스의 item들을 넣기
            target_classes[item[1]] = len(target_classes)
            temp = [] # temp 초기화

        else:
            temp.append(list(item))

        if i == (total_length - 1):
            x.append(temp)


    #print("="*5, "Found {} classes in the feature / {} classes in the target".format(\
    #    len(x), len(target_classes)), "="*5)

    return x, target_classes





class Omniglot_dataset(Dataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        '''
            1) File Import (Images) from root_path 
            2) make a list of all files 
            3) make a dict of classes according to a directory name
            4) __getitem__ plays a role of exporting "img" and "classes" value
                "img" : [filter_count, img_size, img_size]

                self.all_items = [(file_path, directory_name, root_path)] #list
                self.idx_classes = {directory_name, classes} # dict
            
            Args: 
                root_path : root_path of train or test data
                transform : a sequence of transform for an input data
                    e.g)  transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgsz, imgsz)), #resize 명령어는 PIL의 멤버함수
                                                            lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])
                target_transform : a sequence of transform for an target data
                    e.g) None 

            Returns:
                dataset : it contains an omniglot dataset (all images and all classes)
        '''
        
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform

        
        # Make a list of all items (all files)
            #self.all_times : [all_items, 3]
        self.all_items = find_classes(self.root_path)

        # Make a class dict (directory name)
            # self.all_times_classes : [classes, num_point(20), 3]
            # self.idx_classes : [classes, ]
        self.all_items_classes, self.idx_classes = index_classes(self.all_items)

        self.total_length = len(self.all_items)

    def __getitem__(self, idx):
        '''
            returns img, target corresponding the given index

            Args : 
                idx : a row index of dataset

            Returns : 
                img : omniglot image (affected by self.transform, typically np.array)
                target : classes value (sclar, range : [0, 1500])
        '''
        
        
        filename = self.all_items[idx][0]

        # Join filename and root_path
        img_path = str.join('/', [self.all_items[idx][2], filename])

        # Search a key value (all_items[index][1] (directory_name) in the idx_classes (dict))
        target = self.idx_classes[self.all_items[idx][1]] # scalar among [0, 953]

        if self.transform is not None:
            img = self.transform(img_path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
    def __len__(self):
        return (len(self.all_items))


class meta_Omniglot_dataset(Dataset):
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
            2) load the omniglot dataset which consists of all images and target values
                : self.Omniglot_dataset
            3) make a episodes (select tasks). It means a epochs
                : store at self.data called by self._generate_episode
            4) make a each task (select the number of points)
                : in self._generate_eposide, we sample random numbers for task     
    '''
    
    def __init__(self, n_way, k_shot_support, k_shot_query, root_path, img_size, transform=None, types='non_mutual_exclusive'):
        # Img_size
        self.img_size = img_size

        # Transform
            # torch.transform test
        if transform is None:
            self._transform = transforms.Compose([lambda x : Image.open(x).convert('L'),\
                lambda x : x.resize((img_size, img_size)),\
                lambda x : np.array(x)[:,:,None],\
                lambda x : np.transpose(x, (2, 0, 1)),\
                lambda x : x / 255.])
        else:
            self._transform = transform

        # Define Omniglot_dataset
        self.Omniglot_dataset = Omniglot_dataset(root_path, transform=self._transform)

        # Define input and output
        self.inputs, self.outputs = self._get_input_and_target(self.Omniglot_dataset)

        # Define
        self.num_classes = self.inputs.shape[0]
        self.num_points = self.inputs.shape[1]

        # Define Omniglot_dataset
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.types = types

        # Exceptional Treatment
        assert self.k_shot_query + self.k_shot_support <= self.num_points

        # Define episode length
        self.episode_length = self._get_episode_length(self.num_classes, self.n_way)

        # Data
        self.data = self._generate_episode(self.inputs)

    
    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        return self.data[index]
    
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
        perm_num_points = np.random.permutation(self.num_points)
        self.inputs = self.inputs[:, perm_num_points]

        # Random choice of k_shot_support and k_shot_query
        num_point_index = np.random.choice(self.num_points, \
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
            if i + n_way >= num_classes:
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
    
    file_path_1 = "/home/mgyukim/Data/omniglot/images_background"
    file_path_2 = "/home/mgyukim/Data/omniglot/images_evaluation"

    retour = find_classes(file_path_1)
    feature, class_index = index_classes(retour)

    test_file_name = str.join('/', [retour[0][2], retour[0][0]])

    # torch.transform test
    transform = transforms.Compose([lambda x : Image.open(x).convert('L'),\
        lambda x : x.resize((28, 28)),\
        lambda x : np.array(x)])

    x_transform = transforms.Compose([lambda x : Image.open(x).convert('L'),\
        lambda x : x.resize((28, 28)),\
        lambda x : np.array(x)[:,:,None],\
        lambda x : transforms.ToTensor(),\
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    y_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((28, 28)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Test Dataset
    Omniglotdataset = Omniglot_dataset(file_path_1, transform=transform)
    dataset_1 = meta_Omniglot_dataset(5, 5, 5, file_path_1, 28, transform=y_transform, types='inter_shuffle')
    dataset_original = meta_Omniglot_dataset(5, 5, 5, file_path_1, 28, transform=None, types='inter_shuffle')
    
    # Data Loader
    train_dataloader_1 = DataLoader(dataset_1, batch_size=32, shuffle=True, drop_last=True)
    train_dataloader_original = DataLoader(dataset_original, batch_size=32, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader_1):
        support_x, support_y, query_x, query_y = data
        print("torch dataset : ", support_x.shape)

    for i, data in enumerate(train_dataloader_original):
        support_x, support_y, query_x, query_y = data
        print("np dataset : ", support_x.shape)

    
    data2 = dataset[1]

    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)

    


    # Omniglot_dataset test
    
    img, target = Omniglotdataset[0]
    print(img.shape)


    # PIL Image test
    img = Image.open(test_file_name).convert('L')
    img.save('test_image.png')
    img_array = np.array(img) #gray scale 이미지이기 때문에 channel이 없다. 
    

    








    

    

    


    

