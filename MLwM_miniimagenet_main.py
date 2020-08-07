import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Model
from model.maml_meta import Meta
from model.MLwM_model import MLwM

# Dataset
from dataset.MLwM_miniImagenet_dataset import meta_miniImagenet_dataset

# Operator (Trainer, Tester)
from MLwM_operator import MAML_operator

# Config
from MLwM_config import *

# Utils
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
    common.add_argument('--img_size', default=84, type=int, help='image size')
    common.add_argument('--epochs', default=60000, type=int, help='epoch number')
    common.add_argument('--meta_lr', default=1e-3, type=float, help='learning rate for meta update')
    common.add_argument('--update_lr', default=0.01, type=float, help='learning rate for inner update')
    common.add_argument('--update_step', default=5, type=int, help='update steps for meta_training')
    common.add_argument('--update_step_test', default=10, type=int, help='update steps for meta_testing')
    common.add_argument('--encoder_type', default="BBB", type=str, help='what encoder we use')
    common.add_argument('--encoder_output_dim', default=784, type=str, help='output_dim_by_encoded (it should be divided to (img_size * img_size)')
    common.add_argument('--beta_kl', default=1.0, type=float, help='Beta kl')

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

def train(model, save_model_path, initializer=torch.nn.init.xavier_normal_):
    # Create Model
    model = model
    
    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    miniimagenet_training_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, args.img_size, types=args.datatypes)

    miniimagenet_valid_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, args.img_size, mode='val', types=args.datatypes)

    # dataloader
    train_loader = DataLoader(miniimagenet_training_set, batch_size=args.task_size, shuffle=True)
    val_loader = DataLoader(miniimagenet_valid_set, batch_size=1, shuffle=True)

    # Print length of a episode
    print("length of episode : ", len(train_loader))

    if DEBUG:
        support_x, support_y, query_x, query_y = miniimagenet_training_set[0]

        print("="*25, "DEBUG", "="*25)
        print("support_x shape : ", support_x.shape)
        print("support_y shape : ", support_y.shape)
        print("query_x shape : ", query_x.shape)
        print("query_y shape : ", query_y.shape)

    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)

    # Operator
    maml_operator = MAML_operator(model, device, train_loader, optimizer, args.epochs, save_model_path, val_loader)
    maml_operator.train()

    # Save Model
    torch.save(model.state_dict(), os.path.join(save_model_path, "Model_{}.pt".format(args.datatypes)))
    print("="*20, "Save the model (After training)", "="*20)

def test(model, load_model_path, initializer=torch.nn.init.xavier_normal_):
    # Create Model
    model = model
    
    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    miniimagenet_test_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, args.img_size, mode='test', types=args.datatypes)

    test_loader = DataLoader(miniimagenet_test_set, batch_size=1, shuffle=True)

    if DEBUG:
        support_x, support_y, query_x, query_y = miniimagenet_test_set[0]

        print("="*25, "DEBUG", "="*25)
        print("support_x shape : ", support_x.shape)
        print("support_y shape : ", support_y.shape)
        print("query_x shape : ", query_x.shape)
        print("query_y shape : ", query_y.shape)

    # Load a model
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)
    print("="*20, "Load the model : {}".format(load_model_path), "="*20)

    # Operator
    maml_operator = MAML_operator(model, device, test_loader, savedir=save_model_path)
    maml_operator.test()
    

if __name__ == '__main__':
    # DEBUG
    DEBUG = False

    # Parser
    args = parse_args()
    args = set_dir_path_args(args, "miniimagenet")

    # filePath
    miniimagenet_filepath = args.data_path

    # save model path
    save_model_path = get_save_model_path(args)

    # Architecture Config
    if args.model == "MLwM":
        # Set Configuration (Encoder)
        ENCODER_CONFIG = set_config_encoder(ENCODER_CONFIG_MINIIMAGENET, args.encoder_type, args.encoder_output_dim)
        
        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(args.encoder_output_dim))
        architecture = set_config(CONFIG_CONV_4, args.n_way, encoded_img_size, is_regression=False)
    else:
        architecture = set_config(CONFIG_CONV_4_MAXPOOL, args.n_way, args.img_size, is_regression=False)

    # Create Model
    if args.model == "MAML":
        model = Meta(architecture, args.update_lr, args.update_step, is_regression=False)
    elif args.model =="MLwM":
        model = MLwM(ENCODER_CONFIG, architecture, args.update_lr, args.update_step,\
            is_regression=False, is_kl_loss=True, beta_kl=args.beta_kl)
    else:
        NotImplementedError
    
    # Train
    train(model, save_model_path) 

    # load model path
    if args.model_save_root_dir == args.model_load_dir:
        load_model_path = latest_load_model_filepath(args)
    else:
        load_model_path = args.model_load_dir

    # Test
    test(model, load_model_path)

    

    

    



