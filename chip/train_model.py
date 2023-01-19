import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from chip.utils import evaluate_pair

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    training_log = {'epoch': [], 'training_loss': [], 'training_acc':[], 'val_loss': [], 'val_acc': [], 'best_acc': 0.0}

    '''
    Copy an Object in Python
    In Python, we use = operator to create a copy of an object. You may think that this creates a new object; it doesn't. 
    It only creates a new variable that shares the reference of the original object.

    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    threshold = 0.5

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data

            for img0, img1, label in dataloaders[phase]:
                img0 = img0.to(device)
                img1 = img1.to(device)
                label = label.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                is_train = True if(phase == 'train') else False

                with torch.set_grad_enabled(is_train):                    
                    output1, output2 = model(img0, img1)
                    loss = criterion(output1, output2, label)

                    pos_acc, pos_sum, neg_acc, neg_sum = evaluate_pair(output1, output2, label, threshold)
                    # backward + optimize only if in training phase
                    if(phase == 'train'):
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * img0.size(0)                
                running_corrects += pos_acc + neg_acc

            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
            

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
                training_log['training_acc'].append(epoch_acc)            
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                training_log['val_acc'].append(epoch_acc)
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))

    print('Best val Acc: {:4f}'.format(best_acc))
    training_log['best_acc'] = best_acc
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['training_acc'] = np.array(training_log['training_acc'])
    training_log['val_loss'] = np.array(training_log['val_loss'])
    training_log['val_acc'] = np.array(training_log['val_acc'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log