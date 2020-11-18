import os
import subprocess
from datetime import datetime

import numpy as np
import torch 
import umap
import torch.nn
import torch.nn.functional as F
import torchsummary
from torch.utils.tensorboard import SummaryWriter

# Dataset
from dataset.MLwM_miniImagenet_dataset import meta_miniImagenet_dataset

# Model
from model.MAML.maml_meta import Meta

from torch.utils.data import Dataset, DataLoader

from model.MAML.maml_meta import *
from model.LEO.LEO_model import *

from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='MAML for miniimagenet')

    common = parser.add_argument_group('common')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--model', default='MLwM', type=str, help='which model to use ')
    common.add_argument('--datatypes', default='inter_shuffle', type=str, help='which datatype to use')
    common.add_argument('--task_size', default=4, type=int, help='task size')
    common.add_argument('--n_way', default=5, type=int, help='n_way')
    common.add_argument('--k_shot_support', default=1, type=int, help='k shot for support set')
    common.add_argument('--k_shot_query', default=1, type=int, help='k shot for query set')
    common.add_argument('--epochs', default=30000, type=int, help='epoch number')
    common.add_argument('--description', default='Meta_learning', type=str, help='save file name')
    
    # dataset
    common.add_argument('--data_path', default="/home/mgyukim/Data/miniimagenet",\
         type=str, help='directory path for training data')

    # model save dir
    common.add_argument('--model_save_root_dir', default="./save_models/", \
         type=str, help='directory path for test data')

    # model load path
    common.add_argument('--model_load_dir', default="./save_models/", \
         type=str, help='directory path for test data')

    args = parser.parse_args()

    return args

class MAML_plotter(object):
    def __init__(self, model, device, data_loader, on_tensorboard=False):
        # Training setting
        self.device = device
        
        # Data loader
        self.data_loader = data_loader
        
        # Model Import and go to CUDA
        self.model = model
        self.model.to(self.device)
    
        # Model and result Save (I/O)
        self.on_tensorboard = on_tensorboard

    def _epoch_embedded_vectors(self, data_loader):
        # data_loader should call self.dataset.reset_eposide()
        data_loader.dataset.reset_episode()

        # container
        embedded_vectors_list = []
        embedded_vectors_ad_list = []

        for i, data in enumerate(data_loader):
            # Call data
            support_x, support_y, query_x, query_y = data

            # Get shapes
            query_y_shape = query_y.shape
            task_size = query_y_shape[0]
            n_way = query_y_shape[1]
            k_shot = query_y_shape[2]

            # Allocate a device
            support_x = support_x.type(torch.FloatTensor).to(self.device)
            query_x = query_x.type(torch.FloatTensor).to(self.device)

            if self.model._is_regression:
                support_y = support_y.type(torch.FloatTensor).to(self.device)
                query_y = query_y.type(torch.FloatTensor).to(self.device)
            else:
                support_y = support_y.type(torch.LongTensor).to(self.device)
                query_y = query_y.type(torch.LongTensor).to(self.device)

            # Get embedded vectors before task adaptation
            embedded_vectors = self.model.get_embedded_vector_forward(support_x, support_y, query_x, is_adaptation=False)
            embedded_vectors_list.append(embedded_vectors)

            # Get embedded vectors after task adaptation
            embedded_vectors_ad = self.model.get_embedded_vector_forward(support_x, support_y, query_x, is_adaptation=True)
            embedded_vectors_ad_list.append(embedded_vectors_ad)

        # reshape
        embedded_vector_list = torch.stack(embedded_vectors_list, dim=0)
        embedded_vectors_ad_list = torch.stack(embedded_vectors_ad_list, dim=0)

        return embedded_vector_list, embedded_vectors_ad_list

if __name__ == '__main__':
    # DEBUG
    DEBUG = False

    # Parser and set dir_path
    args = parse_args()
    

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # filePath
    miniimagenet_filepath = args.data_path

    # save model path
    save_model_path = get_save_model_path(args)

    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
    config = config['miniImageNet']

    # architecture
    architecture = set_config(config['CONFIG_CONV_4_MAXPOOL'], args.n_way, config['img_size'], is_regression=False)

    # Create Model
    if args.model == "MAML":
        model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=False)

    # load model path
    if args.model_save_root_dir == args.model_load_dir:
        args = set_dir_path_args(args, "miniimagenet")
        load_model_path = latest_load_model_filepath(args)
    else:
        load_model_path = args.model_load_dir

    # dataset
    miniimagenet_valid_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, config['img_size'], mode='val', types=args.datatypes)

    val_loader = DataLoader(miniimagenet_valid_set, batch_size=args.task_size, shuffle=True)

    # Load a model
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)
    print("="*20, "Load the model : {}".format(load_model_path), "="*20)

    # Get vectors
    maml_plotter = MAML_plotter(model, device, val_loader)
    embedded_1, embedded_2 = maml_plotter._epoch_embedded_vectors(val_loader)




    


            
        


        

           