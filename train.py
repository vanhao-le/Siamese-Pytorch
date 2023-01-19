import os
import time
import shutil
import torch
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from create_dataset import ICDAR2011Dataset
import chip.config as config
from model.contrastive_loss import ContrastiveLoss
from model.siamesenet import SiameseNetwork

import matplotlib.pyplot as plt

from chip.train_model import train_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    training_dir = config.training_dir
    training_csv = config.training_csv
    val_dir  = config.val_dir
    val_csv = config.val_csv
    image_size = config.image_size
        
    # Data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomRotation(45),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
           
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),            
        ]),
    }
       
    
    train_dataset = ICDAR2011Dataset(training_csv, training_dir, transform=data_transforms['train'])

    train_loader = DataLoader(train_dataset, 
        batch_size= config.batch_size, shuffle=True, num_workers = config.num_workers
    )

    
    val_dataset = ICDAR2011Dataset(training_csv, training_dir, transform=data_transforms['val'])

    val_loader = DataLoader(val_dataset, 
        batch_size= config.batch_size, shuffle=False, num_workers = config.num_workers
    )

    dataloaders= {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    
    criterion = ContrastiveLoss()
    
    model = SiameseNetwork()    

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_of_gpus)  
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # print(model)

    num_epochs = config.num_epochs
    learning_rate = config.learning_rate

    '''
    What is the problem with all the variant of gradient descent is that it takes lot of time to pass through the gentle slope.
    This is because at gentle slope gradient is very small so update becomes slow.
    Momentum is like a ball rolling downhill. The ball will gain momentum as it rolls down the hill.
    '''
    momentum = config.momentum
    '''
    Weight decay is a regularization technique by adding a small penalty, usually the L2 norm of the weights 
    loss = loss + weight decay parameter * L2 norm of the weights
    '''
    weight_decay = config.weight_decay
    step_size = config.step_size

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,  momentum = momentum, weight_decay= weight_decay)
    '''
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    step_size (int): Period of learning rate decay. It is equivalent to the number of epochs.
    gamma (float): Multiplicative factor of learning rate decay. Default: 0.1 (ten times).
    '''
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs)
    
    df=pd.DataFrame({'epoch':[],'training_loss':[],'training_acc':[],'val_loss':[],'val_acc':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['training_acc'] = log['training_acc']
    df['val_loss'] = log['val_loss']
    df['val_acc'] = log['val_acc']
    df.to_csv(r'output\training_log.csv',columns=['epoch','training_loss','training_acc','val_loss','val_acc'], header=True,index=False,encoding='utf-8')

    model_save_filename = r'output\best_model.pth'
    torch.save(model.state_dict(), model_save_filename)
    '''
    Saving model
    In PyTorch, the learnable parameters (i.e. weights and biases) of an torch.nn.Module model are contained in the model's parameters 
    (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. 
    Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and 
    registered buffers (batchnorm's running_mean) have entries in the model’s state_dict.
    Optimizer objects (torch.optim) also have a state_dict, which contains information about the optimizer's state, 
    as well as the hyperparameters used.

    Because state_dict objects are Python dictionaries, they can be easily saved, updated, altered, and restored, 
    adding a great deal of modularity to PyTorch models and optimizers.
    '''
 

    return



if __name__ == '__main__':
    main()
    
