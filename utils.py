import os
import yaml

import numpy as np
import math

import argparse

# Model
from model.MAML.maml_meta import Meta
from model.MAML.meta_sgd import MetaSGD
from model.MAML.MLwM_model import MLwM
from model.LEO.LEO_model import LEO
from model.SIB.SIB_model import SIB
from model.PrototypicalNet.PrototypicalNet_model import PrototypeNet_embedded
from model.CNP.cnp import CNP

# Helper
from helper.config_helper import *

def parse_args():
    parser = argparse.ArgumentParser(description='MAML for miniimagenet')

    common = parser.add_argument_group('common')
    common.add_argument('--dataset', default='0', type=str, help='which dataset to use')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--model', default='MLwM', type=str, help='which model to use ')
    common.add_argument('--datatypes', default='inter_shuffle', type=str, help='which datatype to use')
    common.add_argument('--task_size', default=4, type=int, help='task size')
    common.add_argument('--task_size_test', default=32, type=int, help='task size')
    common.add_argument('--n_way', default=5, type=int, help='n_way')
    common.add_argument('--k_shot_support', default=1, type=int, help='k shot for support set')
    common.add_argument('--k_shot_query', default=1, type=int, help='k shot for query set')
    common.add_argument('--epochs', default=100000, type=int, help='epoch number')
    common.add_argument('--description', default='embedded_miniimagenet', type=str, help='save file name')
    
    # model save dir
    common.add_argument('--model_root_dir_path', default="./save_models/", \
         type=str, help='directory path for test data')

    args = parser.parse_args()

    return args

def parse_args_test():
    parser = argparse.ArgumentParser(description='Meta learning framework')

    common = parser.add_argument_group('common')
    common.add_argument('--dataset', default='0', type=str, help='which dataset to use')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--model', default='MLwM', type=str, help='which model to use ')
    common.add_argument('--datatypes', default='non_mutual_exclusive', type=str, help='which datatype to use')
    common.add_argument('--task_size_test', default=4, type=int, help='task size')
    common.add_argument('--n_way', default=5, type=int, help='n_way')
    common.add_argument('--k_shot_support', default=5, type=int, help='k shot for support set')
    common.add_argument('--k_shot_query', default=5, type=int, help='k shot for query set')
    common.add_argument('--description', default='embedded_miniimagenet', type=str, help='save file name')
    
    # model load path
    common.add_argument('--model_path', default="./save_models/", \
         type=str, help='.pth path for test data')

    args = parser.parse_args()

    return args


def load_dataset_config(dataset_name, **kwargs):
    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/dataset_config.yml", 'r'), \
                Loader=yaml.SafeLoader)
    
    # Select datasets
    dataset_config = config[dataset_name]

    # Type
    if kwargs.get("type", None) is None:
        dataset_path = dataset_config['data_path']

    elif kwargs.get("type", None) == 'train':
        dataset_path = dataset_config['training_data_path']

    elif kwargs.get("type", None) == 'test':
        dataset_path = dataset_config['test_data_path']

    else:
        NotImplementedError
    
    return dataset_path
    

def load_config_embed_miniimagenet(args):
    '''
        select config files

    '''
    # Initialize
    ENCODER_CONFIG = None
    architecture = None
    
    if args.model == "MLwM":
        # Load config
        config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = config['miniImageNet']
        
        # Set Configuration (Encoder) <- should be modified (stochastic linear 활용)
        ENCODER_CONFIG = set_config_encoder(config['ENCODER_CONFIG_FC'], \
            config['encoder_type_FC'], config['encoder_output_dim'])

        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(config['encoder_output_dim']))
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4'], args.n_way, encoded_img_size, is_regression=False)
        
    elif args.model == "LEO":
        # Config
        leo_config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/LEO_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = leo_config['miniImageNet']

    elif args.model == "SIB":
        # Config
        SIB_config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/SIB_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = SIB_config['miniImageNet']

    elif args.model == "Prototypes_embedded":
        # Config
        Prototypes_embedded_config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/Prototypes_embedded_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = Prototypes_embedded_config['miniImageNet']
    
    else:
        # Set MAML config 
        config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = config['miniImageNet']

        # MetaSGD option
        config["is_meta_sgd"] = True if args.model == "MetaSGD" else False
        
        architecture = set_config_fc_layers(args.n_way, 640, 64, config['layer_count'])

    return config, architecture, ENCODER_CONFIG


def load_config_miniimagenet(args):
    # Initialize
    ENCODER_CONFIG = None
    architecture = None

    # Architecture Config
    if args.model == "MLwM":
        # Load config
        config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
                Loader=yaml.SafeLoader)
        config = config['miniImageNet']
        
        # Set Configuration (Encoder)
        ENCODER_CONFIG = set_config_encoder(config['ENCODER_CONFIG'], \
            config['encoder_type'], config['encoder_output_dim'])
        
        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(config['encoder_output_dim']))
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4'], \
            args.n_way, encoded_img_size, is_regression=False)

    elif args.model == "LEO":
        # Config
        config = yaml.load(open("'/home/mgyukim/workspaces/MLwM/model/LEO/config.yml'", 'r'), \
            Loader=yaml.SafeLoader)
        config = config['miniImageNet']
    else:
        # Load config
        config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
                Loader=yaml.SafeLoader)
        config = config['miniImageNet']

        # MetaSGD option
        config["is_meta_sgd"] = True if args.model == "MetaSGD" else False
        
        # MAML
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4_MAXPOOL'],\
            args.n_way, config['img_size'], is_regression=False)

    return config, architecture, ENCODER_CONFIG


def load_config_omniglot(args):
    # Initialize
    ENCODER_CONFIG = None
    architecture = None
    
    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
    config = config['Omniglot']
    
    # MetaSGD option
    config["is_meta_sgd"] = True if args.model == "MetaSGD" else False

    # Architecture Config
    if args.model == "MLwM":
        # Set Configuration (Encoder)
        ENCODER_CONFIG = set_config_encoder(config['ENCODER_CONFIG'],\
            config['encoder_type'], config['encoder_output_dim'])
        
        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(config['encoder_output_dim']))
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4'], \
            args.n_way, encoded_img_size, is_regression=False)
    else:
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4_MAML'],\
            args.n_way, config['img_size'], is_regression=False)

    return config, architecture, ENCODER_CONFIG


def load_config_poseregression(args):
    # Initialize
    ENCODER_CONFIG = None
    architecture = None
    
    # Load config
    config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/MLwM_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
    config = config['Pose_regression']

    # MetaSGD option
    config["is_meta_sgd"] = True if args.model == "MetaSGD" else False

    # Architecture Config
    if args.model == "MLwM":
        # Set Configuration (MAML)
        encoded_img_size = math.floor(math.sqrt(config['encoder_output_dim']))

        if config['is_image_feature']:
            # Body architecture
            architecture = set_architecture_config_MAML(config['CONFIG_CONV_4'], \
                args.n_way, \
                encoded_img_size,\
                is_regression=True)
        else:
            # Body architecture
            architecture = set_config_fc_layers(1, config['encoder_output_dim'], config['hidden'], config['layer_count'])


    elif args.model == "CNP":
        # Load config
        config = yaml.load(open("/home/mgyukim/workspaces/MLwM/configs/cnp_config.yml", 'r'), \
            Loader=yaml.SafeLoader)
        config = config['Pose_regression']
        architecture = None

    else:
        architecture = set_architecture_config_MAML(config['CONFIG_CONV_4_MAML'],\
            args.n_way, config['img_size'],\
            is_regression=True)

    return config, architecture, ENCODER_CONFIG

def create_model_poseregression(args, config, architecture, ENCODER_CONFIG=None):
    # Create Model
    if args.model == "MAML" or args.model == "MetaSGD":
        # Meta SGD or MAML
        if config['is_meta_sgd']:
            model = MetaSGD(architecture, config['update_lr'], config['update_step'], is_regression=True)
        else:
            model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=True)
    elif args.model =="MLwM":
            model = MLwM(config, architecture, config['update_lr'], config['update_step'],\
                is_regression=True)
    elif args.model == "CNP":
        model = CNP(config, is_regression=True)
    else:
        NotImplementedError

    return model

def create_model_omniglot(args, config, architecture, ENCODER_CONFIG=None):
    # Create Model
    if args.model == "MAML" or args.model == "MetaSGD":
        # Meta SGD or MAML
        if config['is_meta_sgd']:
            model = MetaSGD(architecture, config['update_lr'], config['update_step'], is_regression=False)
        else:
            model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=False)
    elif args.model =="MLwM":
        model = MLwM(ENCODER_CONFIG, architecture, config['update_lr'], config['update_step'], \
            is_regression=False)
    else:
        NotImplementedError

    return model

def create_model_miniimagenet(args, config, architecture, ENCODER_CONFIG=None):
    # Create Model
    if args.model == "MAML" or args.model == "MetaSGD":
        # Meta SGD or MAML
        if config['is_meta_sgd']:
            model = MetaSGD(architecture, config['update_lr'], config['update_step'], is_regression=False)
        else:
            model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=False)

    elif args.model == "LEO":
        model = LEO(config)
        
    elif args.model =="MLwM":
        model = MLwM(ENCODER_CONFIG, architecture, config['update_lr'], config['update_step'],\
            is_regression=False)
    else:
        NotImplementedError

    return model

def create_model_embed_miniimagenet(args, config, architecture, ENCODER_CONFIG=None):
    if args.model == "MAML" or args.model=="MetaSGD":
        # Meta SGD or MAML
        if config['is_meta_sgd']:
            model = MetaSGD(architecture, config['update_lr'], config['update_step'], is_regression=False, is_image_feature=False)
        else:
            model = Meta(architecture, config['update_lr'], config['update_step'], is_regression=False, is_image_feature=False)
    elif args.model == "LEO":
        model = LEO(config)
    elif args.model == "SIB":
        model = SIB(args.n_way, config)
    elif args.model == "Prototypes_embedded":
        model = PrototypeNet_embedded(args.n_way, config)
    elif args.model =="MLwM":
        # MLwM with MAML or MetaSGD
        model = MLwM(ENCODER_CONFIG, architecture, config['update_lr'], config['update_step'],\
            is_regression=False)
    else:
        NotImplementedError

    return model

def output_img_size(img_size, padding, dilation, kernel_size, stride):
    '''
        Figure out feature dimension after Conv Layer 
    '''

        
    numerator = (img_size + (2*padding) - (dilation * (kernel_size - 1)) -1)
    denominator = stride

    length = math.floor((numerator / denominator) + 1)

    if length <= 0:
        print("Conv network is too deep")

    return length

if __name__ == '__main__':
    print(0)

    
    

    

        
    