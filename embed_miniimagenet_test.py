import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset
from torch.utils.data import DataLoader
from dataset.MLwM_embedded_miniimagenet_dataset import meta_embedded_miniimagenet_dataset

# Operator (Trainer, Tester)
from model_operator import model_operator

# Utils
from utils import *
from torchsummary import summary

# Helper
from helper.args_helper import *
from helper.config_helper import *


def test(model, args, config, data_path, load_model_path, save_path, initializer=torch.nn.init.xavier_normal_):    
    # device
    device = torch.device(int(args.device))
    
    # dataset
    miniimagenet_test_set = meta_embedded_miniimagenet_dataset(args.n_way, args.k_shot_support,\
        args.k_shot_query, data_path, mode='train', types=args.datatypes)

    test_loader = DataLoader(miniimagenet_test_set, batch_size=args.task_size_test, shuffle=True)

    # Load a model
    checkpoint = torch.load(load_model_path) 
    model.load_state_dict(checkpoint) 
    print("="*20, "Load the model : {}".format(load_model_path), "="*20)

    # Operator
    maml_operator = model_operator(model, device, test_loader, savedir=save_path)
    maml_operator.test()
    

def embed_miniimagenet_test(args):
    # dataset
    assert args.dataset == "embedded_miniimagenet"

    # filePath
    data_path = load_dataset_config(args.dataset)

    # save model path
    model_dir_path = get_model_dir_path(args, load=True)

    # Get config
    config, architecture, ENCODER_CONFIG = load_config_embed_miniimagenet(args)
        
    # Get Model
    model = create_model_embed_miniimagenet(args, config, architecture, ENCODER_CONFIG)
    
    # Test
    for _ in range(10):
        test(model, args, config, data_path, args.model_path, model_dir_path)

    

    

    



