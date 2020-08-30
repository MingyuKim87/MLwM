import os
import shutil

import numpy as np
import math
from datetime import datetime

import matplotlib.pyplot as plt


def set_config_encoder(config, encoder_type, encoder_output_dim):
    '''
        Set last fc layer should be setted as img_size * img_size

        Args :
            config : encoder configs
            img_size : original img_size

        Return :
            config : encoder configs list
    '''

    # Encoder types
    if encoder_type == "deterministic" or \
        encoder_type == "VAE" or \
            encoder_type == "BBB" or \
                encoder_type == "BBB_FC":

        config[0] = encoder_type
    
    else :
        NotImplementedError

    properties = config[1]
    properties[2] = encoder_output_dim

    config[1] = properties

    return config

def set_config(config, n_way, img_size, is_regression=False):
    '''
        set the architecture of MAML

        e.g)

        CONFIG = [
            ('conv2d', [64, 1, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 2, 2, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('flatten', []),
            ('fc', [N_WAY, 64])
            ]
    '''
    def output_img_size(img_size, padding, dilation, kernel_size, stride):
        numerator = (img_size + (2*padding) - (dilation * (kernel_size - 1)) -1)
        denominator = stride

        length = math.floor((numerator / denominator) + 1)

        return length
    
    for name, properties in config:
        if name == 'flatten':
            return config
        if name == 'conv2d':
            #[ch_out, ch_in, size, size, stride, padding]
            channel = properties[0]
            kernel_size = properties[2]
            stride = properties[4]
            padding = properties[5]
            dilation = 1

            img_size = output_img_size(img_size, padding, dilation, kernel_size, stride)

        if name == "max_pool2d":
            #[size, stride, padding]
            channel = channel
            kernel_size = properties[0]
            stride = properties[1]
            padding = properties[2]
            dilation = 1

            img_size = output_img_size(img_size, padding, dilation, kernel_size, stride)

    
    # figure flatten_size
    flatten_size = channel * img_size * img_size

    # Make flatten layer and fc layer
    if is_regression is not True:
        flatten_layer = ('flatten', [])
        fc_layer = ('fc', [n_way, flatten_size])
    else:
        flatten_layer = ('flatten', [])
        fc_layer = ('fc', [1, flatten_size])

    # Add the last layer
    last_layer = [flatten_layer, fc_layer]

    # Complete a layer
    result = config + last_layer

    return result


def set_config_fc_layers(n_way, embed_size, hidden_size, layer_counts):
    Layers = []

    for i in range(layer_counts):
        if i == 0:
            fc_layer = fc_layer = ('fc', [hidden_size, embed_size])
        elif i == (layer_counts - 1):
            fc_layer = fc_layer = ('fc', [n_way, hidden_size])
        else:
            fc_layer = fc_layer = ('fc', [hidden_size, hidden_size])

        Layers.append(fc_layer)

    return Layers



def get_save_model_path(args):
    # root dir
    save_model_root = args.model_save_root_dir
        
    # Current time
    now = datetime.now()
    current_datetime = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.strftime('%M'))
        
    # Make a save(model) directory (Optional)
    save_model_path = os.path.join(save_model_root, args.datatypes, current_datetime)
    os.makedirs(save_model_path)

    return save_model_path

def latest_load_model_filepath(args):
    # Initialize
    filepath = None
    
    # make a file path
    load_model_path = os.path.join(args.model_load_dir, args.datatypes, "Model_{}.pt".format(args.datatypes))

    # look for the latest directory and model file ~.pt
    if not os.path.isfile(load_model_path):
        temp_path = os.path.join(args.model_load_dir, args.datatypes)
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


def training_line_plot(train_result_file_path, val_result_file_path=None):
    ''' 
        Generate a line plt
        
            Args : 
                training_result_file_path : [num_epochs, 2 (loss, accuracy)]
                val_result_file_path : [num_epochs, 2, (loss, accuracy)]
    '''
    # Get a filename
    temp_dir, _ = os.path.split(train_result_file_path)
    filename = "result_curve_accuracy.png"
    filename = os.path.join(temp_dir, filename)

    # Load a file
    train_result = np.genfromtxt(train_result_file_path, delimiter=',')
    # Slicing a train accuracy
    train_loss = train_result[:, 0]
    train_accuracy = train_result[:, 1]

    if val_result_file_path is not None:
        # Load a data
        val_result = np.genfromtxt(val_result_file_path, delimiter=',')
        val_loss = val_result[:, 0]
        val_accuracy = val_result[:, 1]
    
    # Make a index list
    index_list = np.array(list(range(train_result.shape[0])))
    
    # Make a subplot and decorate this figure
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    # Plot a first figure
    line_1 = ax1.plot(index_list, train_loss, color='tab:red', alpha=0.5, label='train_loss')

    if val_result_file_path is not None:
        line_2 = ax1.plot(index_list, val_loss, color='tab:red', \
            alpha=0.5, linestyle='dashed', label='val_loss')

    ax1.grid()

    if train_accuracy[0] is not None:
        # decorate the second figure
        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy')
        
        line_3 = ax2.plot(index_list, train_accuracy, \
            alpha = 0.5, color='tab:blue', label='train_accuracy')

        if val_result_file_path is not None:
            line_4 = ax2.plot(index_list, val_accuracy, color='tab:blue', \
                alpha=0.5, linestyle='dashed', label='val_accuracy')

    # Decorate a figure
    plt.title('Training & Validation loss and accuracy ', fontsize=20) 
    
    if val_result_file_path is not None:
        if train_accuracy[0] is not None:
            lns = line_1 + line_2 + line_3 + line_4
            plt.legend(lns, ['Train_loss', 'Val_Loss', 'Train_accuracy', 'Val_accuracy'])
        else:
            lns = line_1 + line_2
            plt.legend(lns, ['Train_loss', 'Val_Loss'])

    else:
        if train_accuracy[0] is not None:
            lns = line_1 + line_3
            plt.legend(lns, ['Train_loss', 'Train_accuracy'])
        else:
            lns = line_1
            plt.legend(lns, ['Train_loss'])

    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()


def training_line_plot_regression(train_result_file_path, val_result_file_path=None):
    ''' 
        Generate a line plt
        
            Args : 
                training_result_file_path : [num_epochs, 2 (loss, accuracy)]
                val_result_file_path : [num_epochs, 2, (loss, accuracy)]
    '''
    # Get a filename
    temp_dir, _ = os.path.split(train_result_file_path)
    filename = "result_curve_accuracy.png"
    filename = os.path.join(temp_dir, filename)

    # Load a file
    train_result = np.genfromtxt(train_result_file_path, delimiter=',')
    # Slicing a train accuracy
    train_loss = train_result

    if val_result_file_path is not None:
        # Load a data
        val_result = np.genfromtxt(val_result_file_path, delimiter=',')
        val_loss = val_result
        
    
    # Make a index list
    index_list = np.array(list(range(train_result.shape[0])))
    
    # Plot a first figure
    plt.plot(index_list, train_loss, color='tab:red', alpha=0.5, label='train_loss')

    if val_result_file_path is not None:
        plt.plot(index_list, val_loss, color='tab:red', \
            alpha=0.5, linestyle='dashed', label='val_loss')

    

    # Decorate a figure
    plt.title('Training & Validation loss ', fontsize=15) 

    # Make a subplot and decorate this figure
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Legend
    plt.legend(loc='best')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()

def set_dir_path_args(args, dataset_name):
    model_name = args.model

    model_save_root_path = args.model_save_root_dir
    model_load_root_path = args.model_load_dir

    new_model_save_root_path = os.path.join(model_save_root_path, model_name, dataset_name)
    new_model_load_root_path = os.path.join(model_load_root_path, model_name, dataset_name)

    args.model_save_root_dir = new_model_save_root_path
    args.model_load_dir = new_model_load_root_path

    return args

def remove_temp_files_and_move_directory(save_model_path, result_path, model, \
    encoder_type, beta_kl, problem_name, datatype):
    temp_path = os.path.join(save_model_path, "temp")
    file_names = os.listdir(temp_path)

    for filename in file_names:
        if "20" in filename:
            filepath = os.path.join(temp_path, filename)
            os.remove(filepath)

    last_path = model + "_" + encoder_type + "_" + str(beta_kl) + "_" + problem_name + "_" + datatype
    result_path = os.path.join(result_path, last_path)

    # copy a temp folder to result folder
    shutil.copytree(save_model_path, result_path)
    
    # remove temp folder
    shutil.rmtree(save_model_path)

    print("*"*10, "move the result folder", "*"*10)

    return 0

if __name__ == '__main__':

    # [ch_int, ch_out, kernel_size, kernel_size, stride, padding]
                    # gain=1 according to cbfin's implementation

    CONFIG_OMNIGLOT = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64])
    ]

    CONFIG_IMAGENET = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0])
    ]

    new_config_omniglot = set_config(CONFIG_OMNIGLOT, 5, 28)
    new_config_imagenet = set_config(CONFIG_IMAGENET, 5, 84)

    print(new_config_omniglot)
    print(new_config_imagenet)

    flatten_size_imagenet = 32 * 5 * 5
    print(flatten_size_imagenet)

    a = datetime.now()
    print(a.strftime('%M'))

    training_line_plot("./save_models/MAML/miniimagenet/non_mutual_exclusive/20207312245/temp/result_during_training.txt", \
        "./save_models/MAML/miniimagenet/non_mutual_exclusive/20207312245/temp/val_result_during_training.txt")
    
    

    

        
    