import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True
import numpy as np

import dataset
from models.AlexNet import *
from models.ResNet import *
import os

def save_to_file(name, validation_errors, training_errors, optimizer_name, batch_size, num_epochs, description="no description"):
    to_write = ""
    to_write+="Validation top 5 errors: \n"
    for i in validation_errors:
        to_write+= str(i)+" "
    to_write+="\n"
    to_write+="Traninig top 5 errors: \n"

    for i in training_errors:
        to_write+= str(i)+" "
    to_write+="\n"
    to_write+="optimizer: "+optimizer_name+"\n"
    to_write+="batch size: "+str(batch_size)+"\n"
    to_write+="number of epochs: "+str(num_epochs)+"\n"
    to_write+=description

    with open(os.path.join("./textfiles",name), 'w') as f:
        f.write(to_write)

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
  
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    optimizer_name = "Adam 0.001 "

    epoch = 1
    validation_errors = []
    training_errors = []

    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)
        model.eval() 
        

        correct_val = 0
        total_val = 0
        correct_val_5 = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                _, predicted_5 = torch.topk(outputs.data, 5)
                
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                for i in range (len(labels)):
                    if labels[i] in predicted_5[i,:]:
                        correct_val_5+=1

        print('Validation error of the network TOP 1: %d %%' % (
            100-100 * correct_val / total_val))

        print('Validation error of the network TOP 5: %d %%' % (
            100-100 * correct_val_5 / total_val))
        validation_errors.append(100-100 * correct_val_5 / total_val)
        correct_train = 0
        total_train = 0
        correct_train_5 = 0
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                _, predicted_5 = torch.topk(outputs.data, 5)

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                for i in range (len(labels)):
                    if labels[i] in predicted_5[i,:]:
                        correct_train_5+=1

        print('Train error of the network TOP 1: %d %%' % (
            100-100 * correct_train / total_train))

        print('Train error of the network TOP 5: %d %%' % (
            100-100 * correct_train_5 / total_train))

        training_errors.append(100-100 * correct_train_5 / total_train)
      

        gc.collect()
        epoch += 1
    save_to_file("resnet18adam0_001dropout0_1", validation_errors, training_errors, optimizer_name, batch_size, num_epochs, description="resnet 18 SGD lr 0.001 adam dropout 0.1 ,flipped relu residual,  batch size 100")

print('Starting training')
run()
print('Training terminated')
