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

# Shihouete Coefficient
from sklearn import metrics

# UMAP
import umap

# Dataset
from dataset.MLwM_miniImagenet_dataset import meta_miniImagenet_dataset
from dataset.MLwM_embedded_miniimagenet_dataset import meta_embedded_miniimagenet_dataset

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
    common.add_argument('--em_data_path', default='/home/mgyukim/Data/embeddings/miniImageNet/center',\
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
        """
            Returns:
                embedded_vector_list (num of episodes, task, nway, kshots, 800)
        """
        
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
                #query_y = query_y.type(torch.FloatTensor).to(self.device)
            else:
                support_y = support_y.type(torch.LongTensor).to(self.device)
                #query_y = query_y.type(torch.LongTensor).to(self.device)

            # Get embedded vectors before task adaptation
            embedded_vectors = self.model.get_embedded_vector_forward(support_x, support_y, query_x, is_adaptation=False)
            embedded_vectors_list.append(embedded_vectors)

            # Get embedded vectors after task adaptation
            embedded_vectors_ad = self.model.get_embedded_vector_forward(support_x, support_y, query_x, is_adaptation=True)
            embedded_vectors_ad_list.append(embedded_vectors_ad)

        # reshape
        embedded_vector_list = torch.stack(embedded_vectors_list, dim=0)
        embedded_vectors_ad_list = torch.stack(embedded_vectors_ad_list, dim=0)

        return embedded_vector_list.cpu().detach().numpy(), embedded_vectors_ad_list.cpu().detach().numpy(), query_y.numpy()

    def umap_embedding(self, embedded_vector_list):
        """
           Args:
                embedded_vector_list (num of episodes * task, nway * kshots, 800)

            Returns:
                umap_embedded_vector_list (num of episodes * task, nway * kshots, 2)
        """

        # reducer
        reducer = umap.UMAP(learning_rate=0.5)

        # Container
        umap_embedded_vector_list = []

        # Reducing
        for embedded_vector in embedded_vector_list:
            embedding = reducer.fit_transform(embedded_vector)
            umap_embedded_vector_list.append(embedding)

        return umap_embedded_vector_list

    def umap_embedded_plot_and_score(self, embedding, y, **kwargs):
        """
           Args:
                embedding : (nway * kshots, 2)
                y : (nway * kshots, )
                
            Returns:
                plt.plot
        """
        # Scatter
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y
        )

        # Legend
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))

        # savefig
        if not kwargs.get('filename', None) == None:
            filename = kwargs.get('filename')
            plt.savefig("./results/embedd/" + filename + ".jpg", dpi=300)
        else:
            plt.savefig("./results/embedd/test.jpg", dpi=300)

        plt.clf()

        # Shilouette Coefficient
        score = metrics.silhouette_score(embedding, y, metric='euclidean')

        return score

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

    
    # Permutation index
    #perm_num_points = np.random.permutation(600)
    perm_num_points = np.arange(600)

    # imagenet dataset
    miniimagenet_valid_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, config['img_size'], mode='val', types=args.datatypes, perm_index_list=perm_num_points)

    val_loader = DataLoader(miniimagenet_valid_set, batch_size=args.task_size, shuffle=False)

    # embedded imagenet dataset
    em_miniimagenet_valid_set = meta_embedded_miniimagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.em_data_path, mode='val', types=args.datatypes, perm_index_list=perm_num_points)

    em_val_loader = DataLoader(em_miniimagenet_valid_set, batch_size=args.task_size, shuffle=False)

    em_support_x, em_support_y, em_query_x, em_query_y = iter(em_val_loader).next()

    # Load a model
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)
    print("="*20, "Load the model : {}".format(load_model_path), "="*20)

    # Get vectors
    maml_plotter = MAML_plotter(model, device, val_loader)
    embedded_1, embedded_2, query_y = maml_plotter._epoch_embedded_vectors(val_loader)
    
    em_query_y = em_query_y.numpy()
    
    # Assert
    assert np.array_equal(query_y, em_query_y)

    # Reshaping
        # shape
    episodes, tasks, nways, kshots, dim = embedded_1.shape

    # reshape
    embedded_1 = embedded_1.reshape(episodes*tasks, -1, dim)
    embedded_2 = embedded_2.reshape(episodes*tasks, -1, dim)
    embedded_miniimagenet = em_query_x.reshape(episodes*tasks, nways*kshots, -1)
    query_y = query_y.reshape(episodes*tasks, -1)

    # Reducing dimension
    umap_embedded_before_adaptation = maml_plotter.umap_embedding(embedded_1) # not adaptation
    umap_embedded_after_adaptation = maml_plotter.umap_embedding(embedded_2) # adaptation
    umap_embedded_em_miniimagenet = maml_plotter.umap_embedding(embedded_miniimagenet) #embedded miniimagenet

    # Score append
    score1_list = []
    score2_list = []
    score3_list = []

    for i, (embedd_1, embedd_2, embedd_3, y) in enumerate(zip(umap_embedded_before_adaptation, umap_embedded_after_adaptation, \
        umap_embedded_em_miniimagenet, query_y)):
        # Plot
        score1 = maml_plotter.umap_embedded_plot_and_score(embedd_1, y, filename="maml_before_adaptation_{}".format(i))
        score2 = maml_plotter.umap_embedded_plot_and_score(embedd_2, y, filename="maml_after_adaptation_{}".format(i))
        score3 = maml_plotter.umap_embedded_plot_and_score(embedd_3, y, filename="embedded_miniimagenet_google_{}".format(i))

        # Score
        score1_list.append(score1)
        score2_list.append(score2)
        score3_list.append(score3)

    print(score1_list)
    print(score2_list)
    print(score3_list)



    




    


            
        


        

           