import os
import numpy as np

# Train, Test modules
from embed_miniimagenet_train import embed_miniimagenet_train
from miniimagenet_train import miniimagenet_train
from omniglot_train import omniglot_train
from pose_regression_train import pose_regression_train

# Args
from utils import *

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "embedded_miniimagenet":
        embed_miniimagenet_train(args)
    elif args.dataset == "miniimagenet":
        miniimagenet_train(args)
    elif args.dataset == "Omniglot":
        omniglot_train(args)
    elif args.dataset == "Pose_regression":
        pose_regression_train(args)
    else:
        0

