import os
import glob 

import math
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


class CelebADataset(Dataset):
    '''
        CelebA dataset. 
        PIL packages can load the image as data type "H x W x C"
    '''

    def __init__(self, path_to_data, numsubsample=1, transform=None, mode="train"):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        
        # Set mode
        self.mode = mode
        
        # index
        self.total_count = len(glob.glob(path_to_data + '/*.jpg'))
        self.train_count = int(round(self.total_count * 0.8))
        self.test_count = self.total_count - self.train_count

        # train or test set
        if self.mode == "train" :
            self.img_paths = glob.glob(path_to_data + '/*.jpg')[:self.train_count]
        elif self.mode =="test":
            self.img_paths = glob.glob(path_to_data + '/*.jpg')[self.train_count:]
        else:
            NotImplementedError

        # transform
        self.transform = transform

    def __len__(self):
        if self.mode == "train":
            return self.train_count
        elif self.mode == 'test':
            return self.test_count
        else:
            NotImplementedError
            
    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)

        # Since there are no labels, we just reutn for the 'label' here
        return sample, 0

    def get_img_size(self):
        sample, _ = self.__getitem__(0)
        return sample.size()


    # plotting for celeb A dataset
    def plot_celebA_img(self, all_imgs, is_save = False, file_name='celebA_dataset.png'):
        '''
        
        '''
        # Path
        RESULT_PATH = os.getcwd()
        FILE_NAME = os.path.join(RESULT_PATH, file_name)

        # Img
        img_count = all_imgs.size(0)
        nrow = int(img_count / 2)

        # Plotting
        img_grid = make_grid(all_imgs, nrow=nrow, pad_value = 1.)

        # Visualize
        plt.imshow(img_grid.permute(1, 2,0).numpy())

        # Save the file
        if is_save:
            plt.savefig(FILE_NAME, dpi=300)
        else:
            plt.show()

        plt.clf()


class meta_celebA_dataset(Dataset):
    def __init__(self, k_shot_support, k_shot_query, root_path, x_transform=None, \
        mode=None):

        # Data import
        if x_transform is None:
            self._x_transform = transforms.Compose([
                transforms.CenterCrop(89),
                transforms.Resize(32),
                transforms.ToTensor()
            ])
        else:
            self._x_transform = x_transform

        # mode
        self.mode = mode

        # Set celebA dataset
        self._celebA_dataset = CelebADataset(root_path, transform=self._x_transform, mode=self.mode)

        # Set input and output
        self.imgs, _ = self._get_input_and_target(self._celebA_dataset)

        # Set image_size
        self.channel, self.height, self.width = self._set_img_size()

        # Define the number of classes
        self.num_classes = len(self._celebA_dataset)
        
        # Define celebA_dataset
        self.n_way = 1
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query

        # Define episode length
        self.episode_length = self._get_episode_length(self.num_classes, self.n_way)

        # Make a mask
        self.tasks_support_mask, self.tasks_query_mask = self._tasks_mask(self.num_classes, self.height, self.width, \
            self.k_shot_support, self.k_shot_query)

        # Data
        self.support_xs, self.support_ys, self.query_xs, self.query_ys = self._generate_episode(self.imgs)

    def __len__(self):
        return self.episode_length

    def __getitem__(self, index):
        return self.support_xs[index], self.support_ys[index], self.query_xs[index], self.query_ys[index]

    def _set_img_size(self):
        sample = self.imgs[0]
        channel, height, width = sample.shape

        return channel, height, width

    def _get_episode_length(self, num_classes, n_way):
            return math.floor(num_classes/n_way)

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

        
        inputs = []
        outputs = []

        # Check out the dataset comes from torch.Tensor
        img, label = dataset[0]
        is_torch = torch.is_tensor(img)

        # For container
        for i in range(len(dataset)):
            img, _ = dataset[i]
            inputs.append(img)

        if is_torch:
            # List -> torch.Tensor
            inputs = torch.stack(inputs)
            outputs = torch.LongTensor(outputs)
        else :
            # List -> np.array
            inputs = np.array(inputs)
            outputs = np.array(outputs)

        return inputs, outputs

    def _random_mask(self, height, width, k_shot_support, k_shot_query):
        # Sampling mask index
        measurement = np.random.choice(range(height * width), \
            size=k_shot_support + k_shot_query, replace=False)

        # Create mask container
        support_mask = torch.zeros(height, width).bool()
        query_mask = torch.zeros(height, width).bool()

        # Update mask with mask index
        for i, m in enumerate(measurement):
            row = int(m/width)
            col = m % width

            query_mask[row, col] = 1

            if i < k_shot_support:
                support_mask[row, col] = 1

        return support_mask, query_mask

    def _imgs_masks_to_np_inputs(self, imgs, task_masks, normalize=True):
        '''
            Args
                imgs = (num_total_points(num_tasks), channel, height, width)
                mask = (num_total_points(num_tasks), height, width)

            Returns
                x : unoccluded pixel position [num_total_points(num_tasks), k_shot, 2] 2 means 
                y : maksed img (rbg value) [k_shot, 3]
        '''

        num_tasks, num_channels, height, width = imgs.shape

        # Torch : np.tile
        mask_img_size = task_masks.unsqueeze(dim=1).repeat(1, num_channels, 1, 1)

        # non zero가 아닌 component들의 총 수 
        num_points = task_masks[0].nonzero().size(0)

        # index
        nonzero_idx = task_masks.nonzero() #[num_nonzeros(num_points), 2(height, width)]

        # Positions
        x = nonzero_idx[:, 1:].view(num_tasks, num_points, 2).float()

        # RGB values
        y_temp = imgs[mask_img_size] #[num_tasks, num_channel, height, width] 
        y = y_temp.view(num_tasks, num_channels, num_points) # chennls means 'rgb'
        
        # switch num_channel, num_points --> num_points, num_channel
        y = y.permute(0, 2, 1)

        if normalize:
            x = (x - float(height)/ 2) / (float(height) / 2)
            y -= 0.5

        # Adds n_way dimension
        x = x.unsqueeze(dim=1) #[task_size, n_way, k_shot, 2]
        y = y.unsqueeze(dim=1) #[task_size, n_way, k_shot, 3]

        return x, y

    def _tasks_mask(self, num_classes, height, width, k_shot_support, k_shot_query, is_repeat=True):
        task_context_mask = torch.zeros(num_classes, height, width).bool()
        task_query_mask = torch.zeros(num_classes, height, width).bool()

        if is_repeat:
            support_mask, query_mask = self._random_mask(height, width, k_shot_support, k_shot_query)

            for i in range(num_classes):
                task_context_mask[i] = support_mask
                task_query_mask[i] = query_mask

        else:
            for i in range(num_classes):
                
                support_mask, query_mask = self._random_mask(height, width, k_shot_support, k_shot_query)
                task_context_mask[i] = support_mask
                task_query_mask[i] = query_mask

        return task_context_mask, task_query_mask


    def _generate_episode(self, inputs):
        '''
            Generate a meta-learning episode    

            Args : inputs [num_classes, num_points, channel, height, width]

            Return : a episode
                episode is data set for a epoch (round) [num_task, 4(support_x, support_y, query_x, query_y)]
                    support_x : [n_way, k_shot, 2] -> 2 means x_pos, y_pos
                    support_y : [n_way, k_shot, 3] -> 3 means R, G, B, dim
                    query_x : [n_way, k_shot, 2] -> 2 means x_pos, y_pos
                    query_y : [n_way, k_shot, 3] -> 3 means R, G, B, dim

        '''
        # Dimension check
        num_classes, num_channel, height, width = inputs.shape

        # Make a dataset
        if self.mode == 'train':
            support_xs, support_ys = self._imgs_masks_to_np_inputs(inputs, self.tasks_support_mask)
            query_xs, query_ys = self._imgs_masks_to_np_inputs(inputs, self.tasks_query_mask)

        elif self.mode == 'test':
            support_xs, support_ys = self._imgs_masks_to_np_inputs(inputs, self.tasks_support_mask)
            tasks_query_mask = torch.ones(self.num_classes, self.height, self.width).bool()

            support_xs, support_ys = self._imgs_masks_to_np_inputs(inputs, self.tasks_support_mask)
            query_xs, query_ys = self._imgs_masks_to_np_inputs(inputs, tasks_query_mask)

        else:
            NotImplementedError

        return support_xs, support_ys, query_xs, query_ys

    def reset_episode(self):
        # Regenerate masks
        self.tasks_support_mask, self.tasks_query_mask = \
        self._tasks_mask(self.num_classes, self.height, self.width, self.k_shot_support, self.k_shot_query)

        # Regenerate an eposide
        self.support_xs, self.support_ys, self.query_xs, self.query_ys = self._generate_episode(self.imgs)

        return 0

    def xy_to_img(self, xs, ys):
        '''
            Given an x and y returned by a neural processes, reconstruct images. 
            Missing pixels will have a value of 0.

            Args
            ---------

            xs : torch.Tensor #[task_size, num_points, 2] <- containing normalized indices.  2 means (height, width)
                x \in [0,1] (normalized)
            ys : torch.Tensor #[task_size, num_points, num_channels] where num_channel =1 is grayscale, num_channel = 3 is rgb
            img_size : tuple of int (1, 32, 32) / (3, 32, 32) <- (C, H, W)

            return 
            ---------
                imgs : 
        '''
        # reshapes [num_tasks, n_way, k_shot, 2] --> [num_tasks, k_shot, 2]
        xs = xs.squeeze(dim=1)
        ys = ys.squeeze(dim=1)

        task_size, _, _ = xs.size() # x is torch.Tensor

        # Unnormalize x and y
        xs = xs * float(self.height /2 ) + float(self.height / 2)
        xs = xs.long()

        ys += 0.5

        # Permute y so it match order expected by image 
            # (task_size, num_points, num_channels) --> (task_size, num_channels, num_points)
        ys = ys.permute(0, 2, 1)

        # Initialize empty image

        #img = torch.zeros((batch_size, ) + img_size)
        
        new_img_size = tuple((task_size,)) + tuple((self.channel, self.height, self.width))
        imgs = torch.zeros(new_img_size)
        
        # x[i, :, 0], x[i, :, 1] index를 갖고 있다
        for i in range(task_size):
            imgs[i, :, xs[i, :, 0], xs[i, :, 1]] = ys[i, :, :] # broadcasting for channels

        return imgs


if __name__ == '__main__':
    DEBUG = True


    # Transform
    transform = transforms.Compose([
        transforms.CenterCrop(89),
        transforms.Resize(32),
        transforms.ToTensor()
    ])


    path = '/home/mgyukim/Data/celebA/celebA_img'
    train_dataset = meta_celebA_dataset(100, 100, path, mode='train')
    test_dataset = meta_celebA_dataset(100, 100, path, mode='test')

    if DEBUG:
        train_data = train_dataset[0]
        print(train_data[0].shape)
        print(train_data[1].shape)
        print(train_data[2].shape)
        print(train_data[3].shape)

        test_data = test_dataset[0]

    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for i, data in enumerate(train_dataloader):
        support_x, support_y, query_x, query_y = data

        # Shape
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)

        imgs = train_dataset.xy_to_img(support_x, support_y)
        
        # Img
        img_count = imgs.size(0)
        nrow = int(img_count / 2)

        # Plotting
        img_grid = make_grid(imgs, nrow=nrow, pad_value = 1.)

        # Visualize
        plt.imshow(img_grid.permute(1,2,0).numpy())
        

        # Save the file
        plt.savefig('test.png', dpi=300)
        plt.clf()

        break



    
    for i, data in enumerate(test_dataloader):
        support_x, support_y, query_x, query_y = data
        
        # Shape
        print(support_x.shape)
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)

        imgs = train_dataset.xy_to_img(query_x, query_y)
        
        # Img
        img_count = imgs.size(0)
        nrow = int(img_count / 2)

        # Plotting
        img_grid = make_grid(imgs, nrow=nrow, pad_value = 1.)

        # Visualize
        plt.imshow(img_grid.permute(1,2,0).numpy())
        

        # Save the file
        plt.savefig('test_2.png', dpi=300)
        plt.clf()

        break
    
