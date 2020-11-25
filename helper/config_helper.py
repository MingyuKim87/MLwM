import numpy as np
import math
import yaml

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


def set_architecture_config_MAML(config, n_way, img_size, is_regression=False):
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

        if length <= 0:
            print("Conv network is too deep")

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

    if layer_counts == 1:
        fc_layer = ('fc', [n_way, embed_size])
        Layers.append(fc_layer)

    else:
        for i in range(layer_counts):
            if i == 0:
                fc_layer = ('fc', [hidden_size, embed_size])
            elif i == (layer_counts - 1):
                fc_layer = ('fc', [n_way, hidden_size])
            else:
                fc_layer = ('fc', [hidden_size, hidden_size])

            Layers.append(fc_layer)

    return Layers


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
        ('conv2d', [64, 64, 2, 2, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 1, 1, 1, 0]),
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

    new_config_omniglot = set_config(CONFIG_OMNIGLOT, 5, 14)
    new_config_imagenet = set_config(CONFIG_IMAGENET, 5, 84)

    print(new_config_omniglot)
    print(new_config_imagenet)

    flatten_size_imagenet = 32 * 5 * 5
    print(flatten_size_imagenet)