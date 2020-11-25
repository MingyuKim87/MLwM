import numpy as np
import torch
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='MAML for miniimagenet')

    common = parser.add_argument_group('common')
    

     # tensorboard log path
    common.add_argument('--tensorboard_path', default='./save_models_candidates/',\
         type=str, help='directory path for training data')
         
    # train_result
    common.add_argument('--training_result', default='/home/mgyukim/workspaces/MLwM/save_models_candidates',\
         type=str, help='directory path for training data')

    # val_result
    common.add_argument('--validation_result', default='/home/mgyukim/workspaces/MLwM/save_models_candidates', \
         type=str, help='directory path for test data')

    # test_result
    common.add_argument('--test_result', default='/home/mgyukim/workspaces/MLwM/save_models_candidates', \
         type=str, help='directory path for test data')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
     # Parser
     args = parse_args()
    
     # Writer will output to ./runs/ directory by default
     writer = SummaryWriter(args.tensorboard_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

     # File import
     train_result = np.loadtxt(args.training_result, delimiter=",", dtype=np.float32)
     val_result = np.loadtxt(args.validation_result, delimiter=",", dtype=np.float32)
     test_result = np.loadtxt(args.test_result, delimiter=",", dtype=np.float32)

     for iteration, (test_loss, test_accuracy) in enumerate(test_result):
          writer.add_scalar('Test_loss', test_loss, iteration)
          writer.add_scalar('Test_accuracy', test_accuracy, iteration)

     
     for step, (train_result, val_result) in enumerate(zip(train_result, val_result)):
          writer.add_scalars('Loss', {'train': train_result[0]}, step)
          writer.add_scalars('Accuracy', {'train': train_result[1]}, step)
          writer.add_scalars('Loss', {'val': val_result[0]}, step)
          writer.add_scalars('Accuracy', {'val': val_result[1]}, step)
     
     
     # writer close
     writer.close()

     





