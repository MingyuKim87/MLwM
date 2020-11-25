import os
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Dataset
from dataset.MLwM_omniglot_dataset import meta_Omniglot_dataset

# Operator (Trainer, Tester)
from model_operator import model_operator

# Utils
from utils import *
from torchsummary import summary

# Helper
from helper.args_helper import *
from helper.config_helper import *

def train(model, args, config, train_data_path, test_data_path, save_model_path, \
    initializer=torch.nn.init.xavier_normal_):
    # device
    device = torch.device(int(args.device))

    # dataset
    omniglot_training_set = meta_Omniglot_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        train_data_path, config['img_size'], types=args.datatypes)

    omniglot_valid_set = meta_Omniglot_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        test_data_path, config['img_size'], types=args.datatypes)

    # dataloader
    train_loader = DataLoader(omniglot_training_set, batch_size=args.task_size, shuffle=True)
    valid_loader = DataLoader(omniglot_valid_set, batch_size=args.task_size, shuffle=True)
    
    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['meta_lr'])

    # Operator
    maml_operator = model_operator(model, device, train_loader, optimizer, args.epochs, save_model_path, valid_loader)
    maml_operator.train()

    # Save Model
    torch.save(model.state_dict(), os.path.join(save_model_path, "Model_{}.pt".format(args.datatypes)))
    print("="*20, "Save the model (After training)", "="*20)

    # Move saved files to the result folder
    remove_temp_files_and_move_directory(save_model_path, "/home/mgyukim/workspaces/result_MLwM", args.model, \
        args.dataset, args.datatypes, args.description)
    

def omniglot_train(args):
    # dataset
    assert args.dataset == "Omniglot"

    # Parser (모델별, 데이터타입에 따른 모델 저장 경로 생성)
    args = set_dir_path_args(args, args.dataset, save=True)
    
    # filePath
    train_data_path = load_dataset_config(args.dataset, type='train')
    test_data_path = load_dataset_config(args.dataset, type='test')
    
    # model path
    model_dir_path = get_model_dir_path(args, model_save=True)

    # Get config
    config, architecture, ENCODER_CONFIG = load_config_omniglot(args)

    # Get Model
    model = create_model_omniglot(args, config, architecture, ENCODER_CONFIG)
    
    # Train
    train(model, args, config, train_data_path, test_data_path, model_dir_path) 

    # Close
    print("*"*10, "Finish training", "*"*10)

    

    

    



