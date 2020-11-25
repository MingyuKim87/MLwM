import os
import matplotlib.pyplot as plt

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



if __name__ == '__main__':
    training_line_plot("./save_models/MAML/miniimagenet/non_mutual_exclusive/20207312245/temp/result_during_training.txt", \
            "./save_models/MAML/miniimagenet/non_mutual_exclusive/20207312245/temp/val_result_during_training.txt")