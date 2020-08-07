# "encoder_type" , "img_size", "input_channel", "output_dim", "filter_sizes", "kernel_size"
ENCODER_CONFIG_OMNIGLOT = ['VAE', [28, 1, 784, [32, 48, 64], 3]]
ENCODER_CONFIG_MINIIMAGENET = ['VAE', [84, 3, 784, [32, 48, 64], 3]]
ENCODER_CONFIG_POSE_REGRESSION = ['VAE', [128, 1, 784, [32, 48, 64], 3]]

CONFIG_CONV_4 = [
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

CONFIG_CONV_4_MAXPOOL = [
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

CONFIG_CONV_4_MAXPOOL_ENCODED = [
    ('conv2d', [32, 1, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
]