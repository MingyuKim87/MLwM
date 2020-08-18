import os
from datetime import datetime

import numpy as np
import torch 
import torch.nn
import torch.nn.functional as F
import torchsummary

from torch.utils.data import Dataset, DataLoader

from model.maml_meta import *

from utils import *


class MAML_operator(object):
    def __init__(self, model, device, data_loader, optimizer=None, num_epochs=None, savedir=None, val_data_loader=None):
        # Training setting
        self.optimizer = optimizer
        self.device = device
        
        # Data loader
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        
        # Model
        self.model = model
        self.model.to(self.device)

        # Training parameters
        self.num_epochs = num_epochs
        self.steps = 0 
        self.print_freq = 200
        self.save_freq = 3
        self.figure_freq = 10

        # Model Save (Temp)
        self.savedir = savedir
    
    def _epochs(self, data_loader, train):
        # data_loader should call self.dataset.reset_eposide()
        data_loader.dataset.reset_episode()
        
        epoch_loss = .0
        epoch_criterion = .0
        
        for i, data in enumerate(data_loader):
            # Call data
            support_x, support_y, query_x, query_y = data

            # Shape of query set
            query_y_shape = query_y.shape
            task_size = query_y_shape[0]
            n_way = query_y_shape[1]
            k_shot = query_y_shape[2]
            

            # Allocate a device
            support_x = support_x.type(torch.FloatTensor).to(self.device)
            query_x = query_x.type(torch.FloatTensor).to(self.device)

            if self.model._is_regression:
                support_y = support_y.type(torch.FloatTensor).to(self.device)
                query_y = query_y.type(torch.FloatTensor).to(self.device)

            else:
                support_y = support_y.type(torch.LongTensor).to(self.device)
                query_y = query_y.type(torch.LongTensor).to(self.device)
            
            
            
            # Feed forward
            pred = self.model(support_x, support_y, query_x)

            # loss
            meta_loss, criterion = self.model.meta_loss(support_x, support_y, query_x, query_y)

            if train:
                # Initialization of gradients
                self.optimizer.zero_grad()

                # Calculate the gradients
                meta_loss.backward()

                # Update parameters
                self.optimizer.step()

                # Count steps
                self.steps += 1

                if self.steps % self.print_freq == 1:
                    print("*"*10)
                    print("optimization iteration {}, loss of query set in meta training {:.3f}".format(self.steps, meta_loss.item()))
                    print("optimization iteration {}, criterion of query set in meta training  {:.3f}".format(self.steps, criterion))
                    print("*"*10)

                if self.steps % self.save_freq == 1:
                    # Current Time / Indicate a filename
                    now = datetime.now()
                    currentdate = now.strftime("%Y%m%d%H%M%S")
                    temp_dir = self._make_dir(os.path.join(self.savedir, "temp"))
                    filename = os.path.join(temp_dir, currentdate)
                    
                    # Save Model
                    torch.save(self.model.state_dict(), filename)
                    print("-"*10, "Temporarily save the model", "-"*10)

                
            # Loss
            epoch_loss += meta_loss.item()

            # Crierion (Accuracy or MSE loss)
            if self.model._is_regression:
                pred = pred.view(task_size, n_way, k_shot, -1)
                epoch_criterion += F.mse_loss(pred, query_y)
            else:
                total_elements_in_this_task = torch.numel(query_y)
                epoch_criterion += (torch.eq(query_y, pred).sum().item() / total_elements_in_this_task)

        # Release data and predicted values from cuda
        return (support_x, support_y, query_x, query_y), pred, (epoch_loss, epoch_criterion)

    def train(self):
        
        print('=' * 25, 'Meta trainig', '=' * 25)
        
        for epoch in range(1, self.num_epochs + 1):
            # Run one epoch
            _, _, (epoch_loss, epoch_criterion) = self._epochs(self.data_loader, train=True)
            
            # loss and accuracy should be divided to len(data_loader)
            epoch_loss = epoch_loss / len(self.data_loader)
            epoch_criterion = epoch_criterion / len(self.data_loader)

            # Declare 'None' for valid set
            filename_val_result = epoch_loss_val = epoch_criterion_val = None

            # Result file path
                # Current Time / Indicate a filename
            temp_dir = self._make_dir(os.path.join(self.savedir, "temp"))
            filename_train_result = os.path.join(temp_dir, "result_during_training.txt")
            
            if self.val_data_loader is not None:
                # Run one epoch
                _, _, (epoch_loss_val, epoch_criterion_val) = self._epochs(self.val_data_loader, train=False)

                # loss and accuracy should be divided to len(data_loader)
                epoch_loss_val = epoch_loss_val / len(self.val_data_loader)
                epoch_criterion_val = epoch_criterion_val / len(self.val_data_loader)

                # File path 
                filename_val_result = os.path.join(temp_dir, "val_result_during_training.txt")


            # Write a result 
                # During training
            if not epoch == self.num_epochs:
                # Print a training procedure 
                self._write_results(filename_train_result, epoch, epoch_loss, epoch_criterion, \
                    filename_val_result, epoch_loss_val, epoch_criterion_val)

                # At the last epoch
            else:
                # Current Time / Indicate a filename
                filename_last_result = os.path.join(temp_dir, "train_result.txt")

                # Print and write a file
                self._print_and_write(filename_last_result, '=' * 15 + 'Meta trainig at the last epoch' + '=' * 15)
                self._print_and_write(filename_last_result, '=' * 15 + 'Epoch {} / {}'.format(epoch, self.num_epochs) \
                    + '=' * 15)
                self._print_and_write(filename_last_result, "epoch_loss : {:.3f}".format(epoch_loss / len(self.data_loader)))
                self._print_and_write(filename_last_result, "epoch_criterion : {:.3f}".format(epoch_criterion / len(self.data_loader)))


            # Plot a figure
            if epoch % self.figure_freq == 0:
                training_line_plot(filename_train_result, filename_val_result)
                    
            
    def test(self, update_step=None):
        if update_step is not None:
            self.model.set_update_step(update_step)
        
        # Run one poech
        test_data, pred_y, (_, epoch_criterion) = self._epochs(self.data_loader, train=False)

        # Write the train result
            # Current Time / Indicate a filename
        temp_dir = self._make_dir(os.path.join(self.savedir, "temp"))
        filename = os.path.join(temp_dir, "test_result.txt")

        # Print and write a file
        self._print_and_write(filename, '=' * 25 + 'Meta testing' + '=' * 25)
        self._print_and_write(filename, "1 epoech criterion : {:.3f}".format(epoch_criterion / len(self.data_loader)))


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


if __name__ == "__main__":
    now = datetime.now()

    print(now.strftime("%Y%m%d%H%M%S"))



        
        
        
        
        

            

        


        
        


    


        









            




            
