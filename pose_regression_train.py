import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset
from torch.utils.data import DataLoader
from dataset.MLwM_poseregression_dataset import meta_pose_regression_dataset

# Operator (Trainer, Tester)
from model_operator import model_operator

# Utils
from utils import *
from torchsummary import summary

# Helper
from helper.args_helper import *
from helper.config_helper import *

def train(model, args, config, data_path, save_model_path, initializer=torch.nn.init.xavier_normal_):
    #device
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    #device = torch.device('cuda')
    device = torch.device(int(args.device))

    # dataset
    pose_training_set = meta_pose_regression_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        data_path, types=args.datatypes)

    pose_valid_set = meta_pose_regression_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        data_path, mode='val', types=args.datatypes)

    # dataloader
    train_loader = DataLoader(pose_training_set, batch_size=args.task_size, shuffle=True)
    valid_loader = DataLoader(pose_valid_set, batch_size=args.task_size, shuffle=True)
    
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
    
def pose_regression_train(args):
    # dataset
    assert args.dataset == "Pose_regression"

    # Parser and set dir_path
    args = set_dir_path_args(args, args.dataset, save=True)

    # filePath
    data_path = load_dataset_config(args.dataset)
    
    # model path
    model_dir_path = get_model_dir_path(args, model_save=True)

    # Get config
    config, architecture, ENCODER_CONFIG = load_config_poseregression(args)

    # Get Model
    model = create_model_poseregression(args, config, architecture, ENCODER_CONFIG)
    
    # Train
    train(model, args, config, data_path, model_dir_path) 

    # Close
    print("*"*10, "Finish training", "*"*10)

    

    

    



