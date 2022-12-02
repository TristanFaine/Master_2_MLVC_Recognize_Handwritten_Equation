# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

sourced from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Modified by : Harold Mouchère / University of Nantes

2018

"""
import time
import copy
import os
import torch
import torchvision
import torchvision.transforms as transforms
from random import sample
from collections import Counter

from modules import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.Grayscale(), #CROHME png are RGB, but already 32x32
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

minibatchsize = 8


#TODO: From like 20000 images, get at least "enough" from each class,
#hmmmm
fullset = torchvision.datasets.ImageFolder(root='../data/symbol_recognition', transform=transform)

nb_images = 20_000
partialSet = torch.utils.data.Subset(
    fullset,
    sample(range(nb_images//2), nb_images//2)
)

#split the full train part as train, validation and test, or use the 3 splits defined in the competition
a_part = int(len(partialSet) / 5)
trainset, validationset, testset = torch.utils.data.random_split(partialSet, [3 * a_part, a_part, len(partialSet) - 4 * a_part])

#Get number of samples per class
import numpy as np 

y_train_indices = trainset.indices

y_train = [partialSet[i][1] for i in y_train_indices]

class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

#Then get weights for each class
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)

#Instantiate sampler and remember to refer to it inside the data loader
weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatchsize,
                                        sampler=weighted_sampler,drop_last =True, num_workers=1)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=minibatchsize,
                                        shuffle=False, drop_last =True,num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=minibatchsize,
                                        shuffle=False,drop_last =True, num_workers=0)

# define the set of class names :
classes = [x[0].replace('../data/symbol_recognition/','') for x in os.walk('../data/symbol_recognition/')][1:] # all subdirectories, except itself
print(classes)
print ("nb classes %d , training size %d, val size %d, test size %d" % (len(classes),3*a_part,a_part,len(partialSet) - 4 * a_part ))
########################################################################
# Let us show some of the training images, for fun.
import matplotlib as mpl
mpl.use('Agg') #allows ploting in image without X-server (in docker container)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img, name='output.png'):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#Why do I only have dots and lts??? are they just too prevalent??


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


import torch.nn as nn
import torch.nn.functional as F

########################################################################
# Define the network to use :
net = AlexNet(len(classes))
net.to(device) # move it to GPU or CPU
# show the structure :
print(net)
########################################################################
# Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=3, min_delta=0)
early_stop_break = False

# Definition of arrays to store the results and draw the learning curves
val_err_array = np.array([])
train_err_array = np.array([])
nb_sample_array = np.array([])

# best system results
best_val_loss = 1000000
best_epoch = 0
best_model =  copy.deepcopy(net)

nb_used_sample = 0
running_loss = 0.0
num_epochs = 10
for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # if possible, move them to GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # count how many samples have been used during the training
        nb_used_sample += minibatchsize
        # print/save statistics
        running_loss += loss.item()
        if nb_used_sample % (1000 * minibatchsize) == 0:    # print every 1000 mini-batches
            train_err = (running_loss / (1000 * minibatchsize))
            print('Epoch %d batch %5d ' % (epoch + 1, i + 1))
            print('Train loss : %.3f' % train_err)
            running_loss = 0.0
            #evaluation on validation set
            totalValLoss = 0.0
            with torch.no_grad():
                for data in validationloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    totalValLoss += loss.item()
            val_err = (totalValLoss / len(validationset))
            print('Validation loss mean : %.3f' % val_err)
            train_err_array = np.append(train_err_array, train_err)
            val_err_array = np.append(val_err_array, val_err)
            nb_sample_array = np.append(nb_sample_array, nb_used_sample)

            # save the model only when loss is better
            if val_err <= best_val_loss:
                best_val_loss = val_err
                best_model = copy.deepcopy(net)

            # Early stopping implementation:
            if early_stopper.early_stop(val_err):
                early_stop_break = True   
                break
                
        if early_stop_break: # End iteration over dataloader
            break

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))

    if early_stop_break: # End iterating over epochs
        print("Warning: current epoch stopped early due to early stopping strategy")
        break

print('Finished Training')

### save the best model :
torch.save(best_model.state_dict(), "./segmentReco.nn")

##############################################################################
# Prepare and draw the training curves


plt.clf()
plt.xlabel('epoch')
plt.ylabel('val / train LOSS')
plt.title('Symbol classifier')
plt.plot(nb_sample_array.tolist(), val_err_array.tolist(), 'b',nb_sample_array.tolist(), train_err_array.tolist(), 'r', [best_epoch], [best_val_loss],         'go')
plt.savefig('graph_train_segmentReco.png')

########################################################################
# Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# first on few sample, just to see real results
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)
plt.clf()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(minibatchsize)))
# activate the net with these examples
outputs = best_model(images)

# get the maximum class number for each sample, but print the corresponding class name
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] if predicted[j] in (0, 1) else str(predicted[j])
                              for j in range(minibatchsize)))

# Test now  on the whole test dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# Check the results for each class
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(minibatchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(len(classes)):
    if class_total[i] > 0 :
        print('Accuracy of %5s : %2d %% (%d/%d)' % (
            classes[i], 100 * class_correct[i] / class_total[i], class_correct[i] , class_total[i]))
    else:
        print('No %5s sample' % (classes[i]))




