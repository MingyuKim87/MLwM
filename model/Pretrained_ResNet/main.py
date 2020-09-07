import os
import itertools

import math
from datetime import datetime
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Model
from resnet import WideResNet, get_featnet
from resnet import ClassifierTrain
from resnet import ClassifierEval

# Dataset
from dataset import meta_miniImagenet_dataset, miniImagenet_dataset
from dataset import TrainLoader

# Operator (Trainer, Tester)
from resnet_operator import pretrain_resnet_operator

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
    common.add_argument('--epochs', default=100000, type=int, help='epoch number')
    common.add_argument('--description', default='embedded_miniimagenet', type=str, help='save file name')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    
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

def train(resnet, train_classifier, val_classifier, save_model_path):
    # Create Model
    resnet = resnet
    # 64 : numer of classes in miniimagenet, 640 : number of feature by resent
    train_classifier = train_classifier
    
    # n_way : numer of way, 640 : number of feature by resent
    val_classifier = val_classifier

    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    miniimagenet_training_set = miniImagenet_dataset(args.data_path, mode='train')

    miniimagenet_valid_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, img_size=84, mode='val', types=args.datatypes)

    # dataloader
    train_loader = DataLoader(miniimagenet_training_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(miniimagenet_valid_set, batch_size=args.task_size, shuffle=True)

    # Print length of a episode
    print("length of episode : ", len(train_loader))

    if DEBUG:
        x, y = miniimagenet_training_set[0]

        print("="*25, "DEBUG", "="*25)
        print("x shape : ", x.shape)
        print("y shape : ", y.shape)
        
    # Set the optimizer
    optimizer = torch.optim.SGD(itertools.chain(*[resnet.parameters(),
        train_classifier.parameters()]), args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    # Operator
    resnet_operator = pretrain_resnet_operator(resnet, val_classifier, device, \
        val_loader, optimizer, args.epochs, train_classifier=train_classifier, train_loader=train_loader, savedir=args.model_save_root_dir)

    # train

    
    resnet_operator.train()

    # Save Model
        # 1. resnet
    torch.save(resnet.state_dict(), os.path.join(save_model_path, "resnet.pt"))
    
        # 2. classifer_trainer
    torch.save(train_classifier.state_dict(), os.path.join(save_model_path, "classifier_trainer.pt"))

        # Print
    print("="*20, "Save the model (After training)", "="*20)

    
def test(resnet, val_classifier, load_model_path, save_model_path):
    
    # Create Model
    resnet = resnet
    val_classifier = val_classifier
    
    # Parser
    args = parse_args()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    # dataset
    miniimagenet_test_set = meta_miniImagenet_dataset(args.n_way, args.k_shot_support, args.k_shot_query, \
        args.data_path, img_size=84, mode='test', types=args.datatypes)

    test_loader = DataLoader(miniimagenet_test_set, batch_size=args.task_size, shuffle=True)

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
    resnet_operator = pretrain_resnet_operator(resnet, val_classifier, device, \
        test_loader, optimizer=None, num_epochs=None, train_classifier=None, train_loader=None, savedir=args.model_save_root_dir)

    resnet_operator.test()
    

def set_dir_path_args(args, model_name):
    model_save_root_path = args.model_save_root_dir
    model_load_root_path = args.model_load_dir

    new_model_save_root_path = os.path.join(model_save_root_path, model_name)
    new_model_load_root_path = os.path.join(model_load_root_path, model_name)

    args.model_save_root_dir = new_model_save_root_path
    args.model_load_dir = new_model_load_root_path

    return args

def get_save_model_path(args):
    # root dir
    save_model_root = args.model_save_root_dir
        
    # Current time
    now = datetime.now()
    current_datetime = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.strftime('%M'))
        
    # Make a save(model) directory (Optional)
    save_model_path = os.path.join(save_model_root, args.datatypes, args.description, current_datetime)
    os.makedirs(save_model_path)

    return save_model_path

def latest_load_model_filepath(args):
    # Initialize
    filepath = None
    
    # make a file path
    temp_path = os.path.join(args.model_load_dir)
    item_list = os.listdir(temp_path)
    item_list = sorted(item_list)
    directory_name = item_list[-1]
    
    load_model_path = os.path.join(args.model_load_dir, args.datatypes, args.description,\
         directory_name, "resnet.pt")

    # look for the latest directory and model file ~.pt
    if not os.path.isfile(load_model_path):
        temp_path = os.path.join(args.model_load_dir, args.datatypes, args.description)
        item_list = os.listdir(temp_path)
        item_list = sorted(item_list)

        for item in item_list:
            saved_model_dir = os.path.join(temp_path, item)

            if os.path.isdir(saved_model_dir):
                for f in os.listdir(saved_model_dir):
                    if (f.endswith("pt")):
                        filepath = os.path.join(saved_model_dir, f)
                
        if filepath is None:
            raise NotImplementedError
        else:
            return filepath

    else: 
        return load_model_path


if __name__ == '__main__':
    # DEBUG
    DEBUG = False

    # Parser
    args = parse_args()
    args = set_dir_path_args(args, "ResNet_pretraining")

    # filePath
    miniimagenet_filepath = args.data_path

    # save model path
    save_model_path = get_save_model_path(args)

    # Create Model
    resnet, nFeat = get_featnet('WRN_28_10', 84, 84)
    train_classifier = ClassifierTrain(64, nFeat)
    val_classifier = ClassifierEval(args.n_way, nFeat)
    
    # Train
    train(resnet, train_classifier, val_classifier, save_model_path) 

    # load model path
    if args.model_save_root_dir == args.model_load_dir:
        load_model_path = latest_load_model_filepath(args)
    else:
        load_model_path = args.model_load_dir

    # Test
    test(resnet, val_classifier, load_model_path, save_model_path)

    

    

    



