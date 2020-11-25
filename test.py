import os
import numpy as np

# Train, Test modules
from embed_miniimagenet_test import embed_miniimagenet_test
from miniimagenet_test import miniimagenet_test
from omniglot_test import omniglot_test
from pose_regression_test import pose_regression_test

# Args
from utils import *

if __name__ == "__main__":
    args = parse_args_test()
    
    if args.dataset == "embedded_miniimagenet":
        embed_miniimagenet_test(args)
    elif args.dataset == "miniimagenet":
        miniimagenet_test(args)
    elif args.dataset == "Omniglot":
        omniglot_test(args)
    elif args.dataset == "Pose_regression":
        pose_regression_test(args)
    else:
        0