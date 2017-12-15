"""
Author:     Mitchell Phillips
File:       pytorch_nn_cuda_example.py
Date:       December 2017

Purpose: Pytorch example of training and testing a Fully Connected 
Network and a Convolutional Neural Network. This script was used 
to fulfill course requirements of CSCI 6270 - Computational Vision 
at RPI. However, it is believed that it will be helpful for 
continuing the ISS uNLC project as the next step is to train an FCN. 
This script is intented to aid in the learning process of both CNNs 
as well as torch usage. The purpose of the program is to determine 
the dominant background class of a scene using neural networks. 
The five possibilities considered are grass, wheat-field, road, 
ocean, and red carpet. Pytorch is used to implement two neural 
networks, one using only fully-connected layers, the other using 
convolutional layers in addition to fully-connected layers. 

Resources: Pytorch Documentation and Tutorials. In-class examples. 
http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Note: using 100 character line limit rather than 80.

Usage: The program accepts three command line arguments, an input 
directory, a neural network architecture, and a save/load command. 
Default values have been put in place for these arguments, but is 
recommended to change these. Furthermore, values for the learning 
process, epochs, mini batch size, learning rate, etc., can all be 
modified. Changing the input image size or the network parameters is 
a bit more difficult though. Dimensions need to agree across the 
network and it may be beneficial to check out Stanford's CNN course
notes to learn how what's going on under the hood. 
http://cs231n.github.io/.
"""

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import os
import pickle
import sys
import timeit


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    Assistance from:
    http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    http://pytorch.org/docs/master/nn.html#torch.nn.Sequential
    https://github.com/yunjey/pytorch-tutorial/

    Note: Conv2d(in_channels,out_channels,kernel_size, 
    stride=1, padding=0, dilation=1, groups=1, bias=True)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5, padding=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(2048, 5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x


class FC(nn.Module):
    """
    Fully Connected Neural Network.
    Initial architecture from inclass tutorial.
    """

    def __init__(self, N0, N1, N2, N3, Nout):
        super(FC, self).__init__()
        
        self.fc1 = nn.Linear( N0, N1, bias=True)
        self.fc2 = nn.Linear( N1, N2, bias=True)
        self.fc3 = nn.Linear( N2, N3, bias=True)
        self.fc4 = nn.Linear( N3, Nout, bias=True)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


def directory_valid(directory):
    """
    Validate that the directory is a legitimate location.
    """

    try:
        os.chdir(directory)
    except Exception as e:
        raise argparse.ArgumentTypeError('Please enter a valid directory.')

    return(directory)


def network_valid(nn_architecture):
    """
    Check network architecture decision.
    """
    
    err_message = 'Please enter model as FC or CNN.'
    if nn_architecture != 'FC' and nn_architecture != 'CNN':
        raise argparse.ArgumentTypeError(err_message)   

    return(nn_architecture)


def save_load_valid(save_load):
    """
    Check network architecture decision.
    """
    
    err_message = 'Specify action as save or load.'
    if save_load != 'save' and save_load != 'load' and save_load != False:
        raise argparse.ArgumentTypeError(err_message)   

    return(save_load)


def command_line_parse():
    """
    Parse and check the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-directory',
        help = 'Path to the directory that contains the images to test.',
        default = ('C:\\Users\***EDIT_ME!***'),
        type = directory_valid)
    parser.add_argument('-nn_architecture',
        help = 'Specify to use either an FC or CNN model.',
        default = 'FC',
        type = network_valid)
    parser.add_argument('-save_load',
        help = 'Specify whether or not to save or load data.',
        default = False,
        type = save_load_valid)
    args = parser.parse_args()

    return args


def images_from_directory(directory):
    """
    Find all the image files with '.jpg' or '.JPEG' in the directory.
    INPUT:  directory - Directory containing the images of interest. 
    OUTPUT: img_list - List containing the file names of all the 
            images with '.jpg' or '.JPEG' extension in the directory.
    """

    os.chdir(directory)
    img_list = os.listdir('./')
    img_list = [name for name in img_list if name.lower().endswith(('.jpg','.jpeg'))]
    img_list.sort()

    if len(img_list) < 1:
        print('Not enough images found. End program.')
        sys.exit(0)
    return img_list


def load_images(img_list):
    """
    Load images from a given list. Image axis are swapped from 
    numpy (HxWxC) to torch (CxHxW).
    INPUT:  img_list - List containing file path to images.
    OUTPUT: data - n x C x H x W numpy array of image data. 
    """

    img_size = (30,20) # Reduce image size by a factor of 12.
    data = np.zeros(((len(img_list),3,img_size[1],img_size[0])))

    for j, img in enumerate(img_list):
        img = cv2.imread(img)
        img = cv2.resize(img, img_size)
        data[j,:,:,:] = img.transpose((2, 0, 1))

    return data


def dataloader(train_test_dir):
    """
    Build a the train or test data set given a directory containing 
    either the intended training or testing data.
    In the test or training directory, there should be a list of 
    directors containing the corresponding labeled data. The class 
    labels are derived from these folder titles.
    
    INPUT:  train_test_dir - Directory containing either the training 
            or testing data with appropriately labeled images.
    
    OUTPUT: dataset - Descriptor vectors for each image in their 
            appropriate class.
            targetset - The ground truth labels/classes for each 
            image.
            labels - Labels of training and test data.
    """
   
    os.chdir(train_test_dir)
    label_dir = os.getcwd()
    labels = os.listdir('./')
    labels = [name for name in labels if os.path.isdir(name)]
    labels.sort
    
    dataset = []
    targetset = []

    # Generate  data for k classes.
    for k, target in enumerate(labels):
        img_list = images_from_directory(target)
        img_data = load_images(img_list)
        img_target = [0]*len(labels)
        img_target[k] = 1

        # Data values and labels.
        dataset.extend(img_data)
        targetset.extend([img_target]*len(img_data))
        
        os.chdir(label_dir)
        print('%s complete.' %target)

    # Convert to pytorch tensor variable.
    convert_to_Variable_tensor = [dataset, targetset]
    dataset, targetset = [Variable(torch.FloatTensor(data)) for data in convert_to_Variable_tensor]    
    
    return dataset, targetset, labels


def load_dataset(data_dir):
    """
    Load in the testing and training data to perform 
    classification.
    
    INPUT:  data_dir - Directory containing training and test data.
    
    OUTPUT: train_data - Training data descriptors.
            train_target - Training data ground truth label.
            test_data - Testing data descriptors.
            test_target - Testing data ground truth labels.
            labels - Labels of training and test data.
    """

    os.chdir(data_dir)
    data_list = os.listdir('./')
    data_list.sort()

    if '.ipynb_checkpoints' in data_list:
        data_list.remove('.ipynb_checkpoints')
    
    exit_msg = ('\nDirectory requires both test and train data.')
    
    if len(data_list) != 2 or data_list[1] != 'train' or data_list[0] != 'test':
        print(exit_msg)
        sys.exit(0)
    
    else:
        print('\nGenerating training data...')
        train_data, train_target, labels = dataloader('train')
        os.chdir(data_dir)

        print('\nGenerating testing data...')
        test_data, test_target, labels = dataloader('test')
        os.chdir(data_dir)

    return train_data, train_target, test_data, test_target, labels


def pickle_save(cwd,pickle_data):
    """
    Save data to current working directory.
    INPUT:  cwd - Current working directory.
            pickle_data - Data to be saved.
    OUTPUT: [] - System out. Data saved to directory.
    """

    print('\nSaving data...\n')
    os.chdir(cwd)
    file_Name = 'nn_data.pickle'
    fileObject = open(file_Name,'wb')
    pickle.dump(pickle_data,fileObject)   
    fileObject.close()


def pickle_load(cwd,file_Name):
    """
    Load pickled data from current working directory.
    INPUT:  cwd - Current working directory.
            file_name - Name of pickled data to be loaded.
    OUTPUT: filePickle - Loaded pickle data.
    """

    print('\nLoading data...\n')
    os.chdir(cwd)
    fileObject = open(file_Name,'rb')  
    filePickle = pickle.load(fileObject)

    return filePickle


def import_data(cwd, data_dir, save_load):
    """
    Import data. Either load the data set manually or load from pickled data.
    If loading manually, data may be pickled and saved to current directory 
    if indicated.
    
    INPUT:  cwd - Current working directory.
            data_dir - Directory containing training and test data.
            save_load - Indicate whether to save or load dataset.
    
    OUTPUT: train_data - Training data descriptors.
            train_target - Training data ground truth label.
            test_data - Testing data descriptors.
            test_target - Testing data ground truth labels.
            labels - Labels of training and test data.
    """

    if save_load == 'load': 
        train_data, train_target, test_data, test_target, labels = pickle_load(cwd,'nn_data.pickle')
    else:
        train_data, train_target, test_data, test_target, labels = load_dataset(data_dir)
        if save_load == 'save':
            pickle_save(cwd,[train_data, train_target, test_data, test_target, labels])

    return train_data, train_target, test_data, test_target, labels


def shuffle_data(data, target):
    """
    Shuffle dataset and labels
    INPUT:  data - Dataset.
            target - Dataset labels.
    OUTPUT: data - Shuffled dataset.
            target - Shuffled dataset labels.
    """
    
    n_data = len(target)
    ind_shuffle = np.arange(n_data)
    np.random.shuffle(ind_shuffle)
    ind_shuffle = (torch.from_numpy(ind_shuffle)).type(torch.LongTensor)
    
    data = data[ind_shuffle,:]
    target = target[ind_shuffle,:]

    return data, target


def fc_reshape(convert_to_fc):
    """
    Flatten data in order to provide correct input to fully 
    connected network.
    INPUT:  convert_to_fc - List containing desired data to be flattened.
    OUTPUT: convert_to_fc - List contained flattened data.
    """

    convert_to_fc = [data.view(len(data),data[0].numel()) for data in convert_to_fc]  
    
    return convert_to_fc


def split_train(train_data, train_target):
    """
    Split training data to form training and validation data.
    INPUT:  train_data - Training data descriptors.
            train_target - Training data ground truth label.
    OUTPUT: train_data - Training data descriptors.
            train_target - Training data ground truth label.
            validation_data - Validation data descriptors.
            validation_target - Validation data ground truth label.
    """

    # Split up training data to construct validation data.
    percent_validate = 0.20
    n_train = len(train_target)
    n_validate = int(n_train * percent_validate)

    ind_validate = np.random.choice(n_train,n_validate,replace=False)
    ind_train = np.arange(n_train)
    ind_train = np.delete(ind_train, ind_validate)

    ind_train = (torch.from_numpy(ind_train)).type(torch.LongTensor)
    ind_validate = (torch.from_numpy(ind_validate)).type(torch.LongTensor)

    validate_data = (train_data[ind_validate,:])
    validate_target = (train_target[ind_validate,:])

    train_data = train_data[ind_train,:]
    train_target = train_target[ind_train,:]

    return train_data, train_target, validate_data, validate_target


def initialize_network(nn_architecture, data_shape, Nout):
    """
    Construct neural network to be used, either FC or CNN.
    INPUT:  nn_architecture - Network to be used. FC or CNN
            data_shape - Shape of the input data
            Nout - Number of outputs. Typically the labels. 
    OUTPUT: net - Initialized pytorch neural network.
    """
    
    if nn_architecture=='FC':
        N0 = data_shape[-1]
        N1 = 1250
        N2 = 625
        N3 = 25
        net = FC(N0,N1,N2,N3,Nout)
   
    elif nn_architecture == 'CNN':
        net = CNN()

    params = list(net.parameters())
    print('\nNetwork Architecture:')
    print(net)

    return net


def convert_to_categories(Y):
    """
    Convert pytorch target variables to categories. 
    """
    _, categories = torch.max(Y.data, 1)
    categories = torch.Tensor.long(categories)
    
    return Variable(categories)


def performance_metrics(target_prediction, target, labels):
    """
    Print performance metrics of the network to the screen. Future implementations 
    could include saving loss values and generating plots. 
    INPUT:  target_prediction - Predicted category.
            target - Actual category.
            labels - List of all possible classes. 
    OUTPUT: rate - Overall success rate.
    """

    _,target_prediction_index = torch.max(target_prediction, 1)
    num_equal = torch.sum(target_prediction_index.data == target.data)
    num_different = torch.sum(target_prediction_index.data != target.data)
    rate = num_equal / float(num_equal + num_different)


    # Print individual category accuracy. Rehash of a pytorch tutorial.
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    labels_correct = [0.0] * len(labels)
    labels_total = [0.0] * len(labels)

    label_prediction_correct = (target_prediction_index.data == target.data).squeeze()

    for i in range(len(target_prediction)):
        target_label = (target.data)[i]
        labels_correct[target_label] += label_prediction_correct[i]
        labels_total[target_label] += 1

    for i in range(5):
        print('Accuracy of %5s : %02d %%' % (
            labels[i], 100 * labels_correct[i] / labels_total[i]))


    target = (target.data).tolist()
    target_prediction_index = (target_prediction_index.data).tolist()

    print(classification_report(target, target_prediction_index, target_names=labels))

    # Confusion matrix. Each entry at row r and column c shows 
    # the number of times when r was the correct class label and 
    # c was the chosen class label
    cnf_matrix = confusion_matrix(target, target_prediction_index)
    print('Confusion Matrix:\n %s' % cnf_matrix)
        
    return rate


def train(nn_architecture, train_data, train_target, labels):
    """
    Training neural network.
    INPUT:  nn_architecture - Neural network architecture, FC or CNN.
            train_data, train_target - Training data and data labels.
            labels - Category labels.
    Output: net, criterion - Trained model and loss function.
    """
    
    # Training parameters
    epochs = 1000
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 0.1

    # Split up training data to construct validation data. Convert to categories.
    train_data, train_target, validate_data, validate_target = split_train(train_data, train_target)
    
    # Check cuda support.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('\nUsing cuda.\n')
        train_data, train_target = train_data.cuda(), train_target.cuda()
        validate_data, validate_target = validate_data.cuda(), validate_target.cuda()
        
    train_target = convert_to_categories(train_target)
    validate_target = convert_to_categories(validate_target)

    # Obtain various data dimensions. 
    n_train = len(train_target)
    n_batches = int(np.ceil(n_train / batch_size))
    data_shape= (train_data.data).shape
    Nout = len(labels)

    # Initialize Network
    net = initialize_network(nn_architecture, data_shape, Nout)
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Provide initial prediction.
    validate_target_prediction = net(validate_data)
    valid_loss = criterion(validate_target_prediction, validate_target)
    print("\nInitial Validation Data Loss: %.5f" %valid_loss.data[0])
    
    # Training procedure.
    print('\nBegin training...\n')
    for ep in range(epochs):
        
        # Create a random permutation of the indices.
        indices = torch.randperm(n_train)
        if use_cuda:
            indicies = indices.cuda()
        
        # Run through each mini-batch.
        for b in range(n_batches):
            
            batch_indices = indices[b*batch_size:(b+1)*batch_size]
            if use_cuda:
                batch_indices = batch_indices.cuda()              
           
            batch_data = train_data[batch_indices,:]
            batch_target = train_target[batch_indices]

            optimizer.zero_grad()

            # Forward and backward propagation.
            target_predition = net(batch_data)
            loss = criterion(target_predition, batch_target)

            # Zero the parameter gradients.
            loss.backward()
            optimizer.step()

        target_predition = net(train_data)
        loss = criterion(target_predition, train_target)  
        # Print ongoing statistics.
        if ep != 0 and ep%10==0:
            validate_target_prediction = net(validate_data)
            validate_loss = criterion(validate_target_prediction, validate_target)
            print('Epoch %d:\t Validation Loss: %.5f \t | \t Training Loss: %.5f' %(
                ep, validate_loss.data[0], loss.data[0]))

    print('Finished Training.\n')

    # Final Validation and test loss function values. 
    validate_target_prediction = net(validate_data)
    validate_loss = criterion(validate_target_prediction, validate_target)
    print('\n-----Validation Data Metrics-----\n')
    print('Final validation loss is %.5f' %validate_loss.data[0])

    validate_success_rate = performance_metrics(validate_target_prediction, validate_target, labels)
    print('Validation success rate: %.5f\n' %validate_success_rate)


    # Final training and test loss function values. 
    train_target_prediction = net(train_data)
    loss = criterion(train_target_prediction, train_target)
    print('\n-----Training Data Metrics-----\n')
    print('Final training loss is %.5f' %loss.data[0])

    train_success_rate = performance_metrics(train_target_prediction, train_target, labels)
    print('Training success rate: %.5f\n' %train_success_rate)


    return net, criterion


def test(net, criterion, test_data, test_target, labels):
    """
    Test the performance of the trained neural network.
    INPUT:  net, criterion - Trained model and loss function.
            test_data, test_target -  - Test data and data labels.
            labels - Category labels.
    OUTPUT: [] - System out. Performance metrics printed to screen. 
    """
        
    if torch.cuda.is_available():
        test_data, test_target = test_data.cuda(), test_target.cuda()
    test_target = convert_to_categories(test_target)

    test_target_prediction = net(test_data)
    test_loss = criterion(test_target_prediction, test_target)
    print('\n-----Test Data Metrics-----\n')
    print("Final test loss: %.5f\n" %test_loss.data[0])

    # Classification Success Rate.
    test_success_rate = performance_metrics(test_target_prediction, test_target, labels)
    print('Test success rate: %.5f' %test_success_rate)


def main():
    """
    INPUT:  sys.argv[1] - Path to the directory containing the 
            training and testing data.
    OUTPUT: [] - System out. Print performance statistics to command line.
    """
    
    # System inputs.
    nn_start = timeit.default_timer()
    cwd = os.getcwd()
    args = command_line_parse()
    data_dir = args.directory
    nn_architecture = args.nn_architecture
    save_load = args.save_load
    
    # Load and shuffle data.
    train_data, train_target, test_data, test_target, labels = import_data(cwd, data_dir, save_load)
    train_data, train_target = shuffle_data(train_data, train_target)
    test_data, test_target = shuffle_data(test_data, test_target)

    # Reshape data if using FC. Network requires flattened inputs.
    if nn_architecture == 'FC':     
        train_data, train_target, test_data, test_target = fc_reshape(
            [train_data, train_target, test_data, test_target])

    # Train and test network.
    net, criterion = train(nn_architecture, train_data, train_target, labels)
    test(net, criterion, test_data, test_target, labels)

    nn_stop = timeit.default_timer()
    nn_runtime = nn_stop-nn_start
    print('\nTotal Runtime: %.2f secs.' %nn_runtime)
    

if __name__ == '__main__':
    main()