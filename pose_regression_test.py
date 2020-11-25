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

def test(model, args, data_path, load_model_path, save_model_path, initializer=torch.nn.init.xavier_normal_):
    # device
    device = torch.device(int(args.device))
    
    # dataset
    pose_test_set = meta_pose_regression_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        data_path, mode='val', types=args.datatypes)

    test_loader = DataLoader(pose_test_set, batch_size=args.task_size_test, shuffle=True)

    # Load a model
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)
    print("="*20, "Load the model : {}".format(load_model_path), "="*20)

    # Operator
    maml_operator = model_operator(model, device, test_loader, savedir=save_model_path)
    maml_operator.test()
    

def pose_regression_test(args):
    # dataset
    assert args.dataset == 'Pose_regression'

    # Parser
    args = parse_args_test()

    # filePath
    data_path = load_dataset_config(args.dataset)
    
    # save model path
    model_dir_path = get_model_dir_path(args, load=True)

    # Get config
    config, architecture, ENCODER_CONFIG = load_config_poseregression(args)

    # Get Model
    model = create_model_poseregression(args, config, architecture, ENCODER_CONFIG)

    # Test
    for _ in range(10):
        test(model, args, data_path, args.model_path, model_dir_path)

    

    

    



