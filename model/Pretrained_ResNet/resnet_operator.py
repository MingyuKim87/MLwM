import itertools
import os
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from criterion import accuracy, AverageMeter

from resnet import get_featnet
from dataset import TrainLoader
from dataset import meta_miniImagenet_dataset

import matplotlib.pyplot as plt

randomSeed = 123
torch.backends.cudnn.deterministic = True
torch.manual_seed(randomSeed)

class pretrain_resnet_operator(object):
    def __init__(self, resnet, val_classifer, device, \
        val_loader, optimizer=None, num_epochs=None, train_classifier=None, \
        train_loader=None, savedir=None, lr_decay_milestone=[50]):

        # Training setting
        self.optimizer = optimizer
        self.lrScheduler = MultiStepLR(self.optimizer, \
            milestones=lr_decay_milestone, gamma=0.1)

        # Device
        self.device = device
        
        # Data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Model
        self.resnet = resnet
        self.train_classifier = train_classifier
        self.val_classifier = val_classifer

        # To device
        self.resnet.to(self.device)
        self.train_classifier.to(self.device)
        self.val_classifier.to(self.device)

        # Criterion (train loss)
        self.criterion = F.cross_entropy

        # Training parameters
        self.num_epochs = num_epochs
        self.steps = 0 
        self.print_freq = 200
        self.save_freq = 1000
        self.figure_freq = 3000

        # Model Save (Temp)
        self.savedir = savedir


    def _train_epochs(self, train_loader):
        # Eval model
        self.resnet.train()
        self.train_classifier.train()
        
        # Loss
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, data in enumerate(train_loader):
            # Call data
            x, y = data

            # Allocate a device
            x = x.type(torch.FloatTensor).to(self.device)
            y = y.type(torch.LongTensor).to(self.device)

            # Feed forward
            pred = self.train_classifier(self.resnet(x))

            # loss
            loss = self.criterion(pred, y, reduction='sum')

            # Initialization of gradients
            self.optimizer.zero_grad()

            # Calculate the gradients
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Count steps
            self.steps += 1

            # accruacy
            acc1, acc5 = accuracy(pred, y, topk=(1, 5))

            losses.update(loss.item(), x.size()[0])
            top1.update(acc1[0].item(), x.size()[0])
            top5.update(acc5[0].item(), x.size()[0])

            # Printing
            if self.steps % self.print_freq == 1:
                print("*"*10)
                print("optimization iteration {}, loss {:.3f}".format(self.steps, losses.avg))
                print("optimization iteration {}, accuracy 1 / accuracy 5 {:.3f}, {:.3f}"\
                    .format(self.steps, top1.avg, top5.avg))
                print("*"*10)

            if self.steps % self.save_freq == 1:
                # Current Time / Indicate a filename
                now = datetime.now()
                currentdate = now.strftime("%Y%m%d%H%M%S")
                temp_dir = self._make_dir(os.path.join(self.savedir, "resnet"))
                filename = os.path.join(temp_dir, currentdate)
                
                # Save Model
                torch.save(self.resnet.state_dict(), filename + "_resnet.pt")
                torch.save(self.train_classifier.state_dict(), filename + "_classifier.pt")
                print("-"*10, "Temporarily save the model", "-"*10)

        return losses.avg, top1.avg, top5.avg


    def _test_epochs(self, val_loader):
        # Eval model
        self.resnet.eval()
        self.val_classifier.eval()

        # Metric
        top1 = AverageMeter()

        for i, data in enumerate(self.val_loader):
            # Call data
            support_x, support_y, query_x, query_y = data

            # Get shape
            task_size, n_way, k_shot_support, channel, height, width = support_x.size()
            _, _, k_shot_query, _, _, _ = query_x.size()

            # Allocate a device
            support_x = support_x.type(torch.FloatTensor).to(self.device)
            support_y = support_y.type(torch.LongTensor).to(self.device)
            
            query_x = query_x.type(torch.FloatTensor).to(self.device)
            query_y = query_y.type(torch.LongTensor).to(self.device)

            # Feed forward
                # Embedding by resnet

            with torch.no_grad():
                support_x = support_x.view(-1, channel, height, width)
                query_x = query_x.view(-1, channel, height, width)

                # Embedding
                support_embedded_x = self.resnet(support_x) #[task_size, n_way, k_shot, embedded_dim]
                query_embedded_x = self.resnet(query_x) #[task_size, n_way, k_shot, embedded_dim]

                # Reshape embedded_xs
                support_embedded_x = support_embedded_x.view(task_size, n_way, k_shot_support, -1) #[task_size, n_way, k_shot, embedded_dim]
                query_embedded_x = query_embedded_x.view(task_size, (n_way * k_shot_query), -1) #[task_size, n_way * k_shot, embedded_dim]
                query_y = query_y.view(task_size, -1)

                # Validation Classifier (Cosine similarity)
                cosine_similarity = self.val_classifier(support_embedded_x, query_embedded_x) #[task_size, n_way * k_shot, n_way]
            
            # Evaluate
            acc1 = accuracy(cosine_similarity, query_y, topk=(1, ))
            top1.update(acc1[0].item(), cosine_similarity.size()[0])

        return top1.avg

    def train(self):
        # Declare 'None' for valid set
        filename_val_result = epoch_loss_val = epoch_criterion_val = None

        # Result file path
            # Current Time / Indicate a filename
        temp_dir = self._make_dir(os.path.join(self.savedir, "resnet_pretraining"))
        filename_train_result = os.path.join(temp_dir, "result_during_training.txt")

        print('=' * 25, 'Warm UP', '=' * 25)
        self.LrWarmUp(6000, lr)
            
        print('=' * 25, 'Meta trainig', '=' * 25)

        for epoch in range(1, self.num_epochs + 1):    
            # Run one epoch
            epoch_losses, epoch_top1, epoch_top5 = self._train_epochs(self.train_loader)

            # update_lr
            self.lrScheduler.step()

            if self.val_loader is not None:
                # Run one epoch
                with torch.no_grad():
                    epoch_top1_val = self._test_epochs(self.val_loader)

            # Write a result 
                # During training
            if not epoch == self.num_epochs:
                # Print a training procedure 
                self._write_results(filename_train_result, epoch, epoch_losses, epoch_top1, \
                    filename_val_result, 0, epoch_top1_val)

            else:
                # Current Time / Indicate a filename
                filename_last_result = os.path.join(temp_dir, "train_result.txt")

                # Print and write a file
                self._print_and_write(filename_last_result, '=' * 15 + 'Meta trainig at the last epoch' + '=' * 15)
                self._print_and_write(filename_last_result, '=' * 15 + 'Epoch {} / {}'.format(epoch, self.num_epochs) \
                    + '=' * 15)
                self._print_and_write(filename_last_result, "epoch_loss : {:.3f}".format(epoch_losses))
                self._print_and_write(filename_last_result, "epoch_top1 : {:.3f}".format(epoch_top1))
                self._print_and_write(filename_last_result, "epoch_top5 : {:.3f}".format(epoch_top5))

            # Plot a figure
            if epoch % self.figure_freq == 0:
                self.training_line_plot(filename_train_result, filename_val_result)

    
    def test(self):
        # Run one poech
        with torch.no_grad():
            epoch_top1_val = self._test_epochs(self.val_loader)

        # Write the train result
            # Current Time / Indicate a filename
        temp_dir = self._make_dir(os.path.join(self.savedir, "temp"))
        filename = os.path.join(temp_dir, "test_result.txt")

        # Print and write a file
        self._print_and_write(filename, '=' * 25 + 'Meta testing' + '=' * 25)
        self._print_and_write(filename, "1 epoech criterion : {:.3f}".format(epoch_top1_val))

    def LrWarmUp(self, total_iteration, lr):
        iteration = 0
        updated_lr = lr
        valTop1 = 0

        while iteration < total_iteration :
            self.resnet.train()
            self.train_classifier.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for i, data in enumerate(self.train_loader):
                # Call data
                x, y = data

                # Allocate a device
                x = x.type(torch.FloatTensor).to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                
                
                iteration += 1

                if iteration == total_iteration:
                    break

                updated_lr = iteration / float(total_iteration) * lr
                
                for g in self.optimizer.param_groups:
                    g['lr'] = updated_lr

                self.optimizer.zero_grad()

                pred = self.train_classifier(self.resnet(x))
                loss = self.criterion(pred, y, reduction='sum')

                loss.backward()
                self.optimizer.step()

                acc1, acc5 = accuracy(pred, y, topk=(1, 5))

                losses.update(loss.item(), x.size()[0])
                top1.update(acc1[0].item(), x.size()[0])
                top5.update(acc5[0].item(), x.size()[0])

                print("*"*10)
                print("optimization iteration {}, loss {:.3f}".format(self.steps, losses.avg))
                print("optimization iteration {}, accuracy 1 / accuracy 5 {:.3f}, {:.3f}"\
                    .format(self.steps, top1.avg, top5.avg))
                print("*"*10)

            with torch.no_grad():
                valTop1 = self._test_epochs(self.val_loader)

            return valTop1

    def _make_dir(self, dirpath):
        try:
            if not(os.path.isdir(dirpath)):
                os.makedirs(os.path.join(dirpath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
        
        return os.path.join(dirpath)

    def _print_and_write(self, filepath, string):
        if not os.path.isfile(filepath):
            f = open(filepath, "w")
        else:
            f = open(filepath, 'a')
        f.write(string + "\n")
        print(string)

    def _write_results(self, filepath, epoch, epoch_loss, epoch_criterion, \
            filepath2=None, epoch_loss_val=None, epoch_criterion_val=None):
        # Print a training procedure 
        print('=' * 25, 'Epoch {} / {}'.format(epoch, self.num_epochs), '=' * 25)
        print("epoch_loss : {:.3f}".format(epoch_loss))
        print("epoch_criterion : {:.3f}".format(epoch_criterion))

        with open(filepath, 'ab') as f:
            epoch_result = [[epoch_loss, epoch_criterion]]
            np.savetxt(f, epoch_result, delimiter=',', fmt='%.3f')
        
        
        if filepath2 is not None:
            print("epoch_loss_val : {:.3f}".format(epoch_loss_val))
            print("epoch_criterion_val : {:.3f}".format(epoch_criterion_val))

            with open(filepath2, 'ab') as f:
                epoch_result = [[epoch_loss_val, epoch_criterion_val]]
                np.savetxt(f, epoch_result, delimiter=',', fmt='%.3f')

    def training_line_plot(self, train_result_file_path, val_result_file_path=None):
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
        train_accuracy1 = train_result[:, 1]
        train_accuracy2 = train_result[:, 1]

        if val_result_file_path is not None:
            # Load a data
            val_result = np.genfromtxt(val_result_file_path, delimiter=',')
            _ = val_result[:, 0]
            val_accuracy = val_result[:, 1]
        
        # Make a index list
        index_list = np.array(list(range(train_result.shape[0])))
        
        # Make a subplot and decorate this figure
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        
        # Plot a first figure
        line_1 = ax1.plot(index_list, train_loss, color='tab:red', alpha=0.5, label='train_loss')
        ax1.grid()

        if train_accuracy1[0] is not None:
            # decorate the second figure
            ax2 = ax1.twinx()
            ax2.set_ylabel('accuracy')
            
            line_3 = ax2.plot(index_list, train_accuracy1, \
                alpha = 0.5, color='tab:blue', label='train_top1')

            line_5 = ax2.plot(index_list, train_accuracy2, \
                alpha = 0.5, color='tab:navy', label='train_top5')

            if val_result_file_path is not None:
                line_4 = ax2.plot(index_list, val_accuracy, color='tab:blue', \
                    alpha=0.5, linestyle='dashed', label='val_top1')

        # Decorate a figure
        plt.title('Training & Validation loss and accuracy ', fontsize=20) 
        
        if val_result_file_path is not None:
            if train_accuracy1[0] is not None:
                lns = line_1 + line_3 + line_4 + line_5
                plt.legend(lns, ['Train_loss', 'Train_top1', 'Train_top5', 'Val_top1'])
            else:
                lns = line_1
                plt.legend(lns, ['Train_loss'])

        else:
            if train_accuracy1[0] is not None:
                lns = line_1 + line_3 + line_4
                plt.legend(lns, ['Train_loss', 'Train_top1', 'Train_top5'])
            else:
                lns = line_1
                plt.legend(lns, ['Train_loss'])

        plt.tight_layout()

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.clf()


                    

        



                

                
                
















        def _make_dir(self, dirpath):
            try:
                if not(os.path.isdir(dirpath)):
                    os.makedirs(os.path.join(dirpath))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!!!!!")
            
            return os.path.join(dirpath)

        





            
           

        

