import os
import yaml


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Model
from model.MAML.maml_meta import Meta
from model.MAML.MLwM_model import MLwM

# Dataset
from dataset.MLwM_omniglot_dataset import meta_Omniglot_dataset
from dataset.MLwM_miniImagenet_dataset import meta_miniImagenet_dataset

# Operator (Trainer, Tester)
from MLwM_operator import MAML_operator

# Utils
from utils import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MAML for omniglot')

    common = parser.add_argument_group('common')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--model', default='MLwM', type=str, help='which model to use ')
    common.add_argument('--datatypes', default='inter_shuffle', type=str, help='which datatype to use')
    common.add_argument('--task_size', default=16, type=int, help='task size')
    common.add_argument('--n_way', default=5, type=int, help='n_way')
    common.add_argument('--k_shot_support', default=1, type=int, help='k shot for support set')
    common.add_argument('--k_shot_query', default=1, type=int, help='k shot for query set')
    common.add_argument('--epochs', default=60000, type=int, help='epoch number')
    common.add_argument('--description', default='Meta_learning', type=str, help='save file name')
    
    # dataset
    common.add_argument('--training_data_path', default="/home/mgyukim/Data/omniglot/images_background",\
         type=str, help='directory path for training data')

    common.add_argument('--test_data_path', default="/home/mgyukim/Data/omniglot/images_evaluation", \
         type=str, help='directory path for test data')

    # model save dir
    common.add_argument('--model_save_root_dir', default="./save_models/", \
         type=str, help='directory path for test data')

    # model load path
    common.add_argument('--model_load_dir', default="./save_models/", \
         type=str, help='directory path for test data')


    args = parser.parse_args()

    return args

def train(model, config, save_model_path, initializer=torch.nn.init.xavier_normal_):
    # Create Model
    model = model
    
    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    omniglot_training_set = meta_Omniglot_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.training_data_path, config['img_size'], types=args.datatypes)

    omniglot_valid_set = meta_Omniglot_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.test_data_path, config['img_size'], types=args.datatypes)

    # dataloader
    train_loader = DataLoader(omniglot_training_set, batch_size=args.task_size, shuffle=True)
    valid_loader = DataLoader(omniglot_valid_set, batch_size=args.task_size, shuffle=True)
    

    # Print length of a episode
    print("length of episode : ", len(train_loader))

    if DEBUG:
        support_x, support_y, query_x, query_y = omniglot_training_set[0]

        print("="*25, "DEBUG", "="*25)
        print("support_x shape : ", support_x.shape)
        print("support_y shape : ", support_y.shape)
        print("query_x shape : ", query_x.shape)
        print("query_y shape : ", query_y.shape)

    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['meta_lr'])

    # Operator
    maml_operator = MAML_operator(model, device, train_loader, optimizer, args.epochs, save_model_path, valid_loader)
    maml_operator.train()

    # Save Model
    torch.save(model.state_dict(), os.path.join(save_model_path, "Model_{}.pt".format(args.datatypes)))
    print("="*20, "Save the model (After training)", "="*20)

    
    # Move saved files to the result folder
    remove_temp_files_and_move_directory(save_model_path, "/home/mgyukim/workspaces/result_MLwM", args.model, \
        config['encoder_type'], config['beta_kl'], "Omniglot", args.datatypes, args.description)
    

def test(model, config, load_model_path, save_model_path, initializer=torch.nn.init.xavier_normal_):
    # Create Model
    model = model
    
    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    omniglot_test_set = meta_Omniglot_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.test_data_path, config['img_size'], types=args.datatypes)

    test_loader = DataLoader(omniglot_test_set, batch_size=args.task_size, shuffle=True)

    if DEBUG:
        support_x, support_y, query_x, query_y = omniglot_test_set[0]

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
    DEBUG = True

    # Parser
    args = parse_args()
    args = set_dir_path_args(args, "Omniglot")
    
    # Data Path
    omniglot_training_filepath = args.training_data_path
    omniglot_test_filepath = args.test_data_path
    
    # save model path
    save_model_path = get_save_model_path(args)

    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
    config = config['Omniglot']
    
    # Architecture Config
    if args.model == "MLwM":
        # Set Configuration (Encoder)
        ENCODER_CONFIG = set_config_encoder(config['ENCODER_CONFIG'],\
            config['encoder_type'], config['encoder_output_dim'])
        
        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(config['encoder_output_dim']))
        architecture = set_config(config['CONFIG_CONV_4'], args.n_way, encoded_img_size, is_regression=False)
    else:
        architecture = set_config(config['CONFIG_CONV_4_MAML'], args.n_way, config['img_size'], is_regression=False)

    # Create Model
    if args.model == "MAML":
        model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=False)
    elif args.model =="MLwM":
        model = MLwM(ENCODER_CONFIG, architecture, config['update_lr'], config['update_step'], \
            is_regression=False, is_kl_loss=True, beta_kl=config['beta_kl'])
    else:
        NotImplementedError
    
    # Train
    train(model, config, save_model_path) 

    # load model path
    if args.model_save_root_dir == args.model_load_dir:
        load_model_path = latest_load_model_filepath(args)
    else:
        load_model_path = args.model_load_dir

    # Test
    test(model, config, load_model_path, save_model_path)

    

    

    



