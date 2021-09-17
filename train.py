from __future__ import print_function, division
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import csv
from pathlib import Path
from torchvision.models import mobilenet_v2


def get_n_image_per_classes(dataset):
    """
    get_n_image_per_classes gets the csv file containing 
    informations about dataset, and return the number of 
    images for each class.

    :param dataset: csv file containing images names and associated labels.
    :return: dict with class name as keys and number of image associated as values.
    """ 
    image_classs_dict = {}
    with open(dataset, newline='') as csvfile:
        #We skip the first line as it contains header
        next(csvfile)
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # For each row of the csv file we retrieve the class name
            # and the path to our image
            class_name = row[1]
            image_path = 'data/images/' + str(row[0]).split("/")[1]
            # Sometimes our csv may contains images that are not present in our dataset
            # so we have to first check if they exist or not to avoid errors
            if (os.path.isfile(image_path)):
                if (class_name not in image_classs_dict):
                    image_classs_dict[class_name] = 1
                else:
                    image_classs_dict[class_name] += 1
    return image_classs_dict

def get_split_size(n_image_per_classes):
    """
    get_split_size gets a dict with class name as keys and number of image associated as values.
    it then proceed to compute 80% of each class number of images for the training set, 20% will be used for validation.

    :param n_image_per_classes: dict with class name as keys and number of image associated as values.
    :return: dict with class name as keys and number of image associated for training as values.
    """ 
    for key in n_image_per_classes:
        # We want 80% of each class for training, and 20% for validation
        n_image_per_classes[key] = round(n_image_per_classes[key] * 0.8)
    return n_image_per_classes

def preprocess_data(dataset):
    """
    preprocess_data gets gets the csv file and process data 
    by splittiing it into two folders for training and validation and removing missing values.

    :param dataset: csv file containing images names and associated labels.
    :return: nothing.
    """ 
    class_list = set()
    n_image_per_classes = get_n_image_per_classes(dataset)
    split_size = get_split_size(n_image_per_classes)
    with open(dataset, newline='') as csvfile:
        next(csvfile)
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            # We get the class number
            class_name = row[1]
            image_name = row[0]
            image_path = 'data/images/' + str(image_name).split('/')[1]
            # For each class we create a folder that will contain our train/validation images
            if (os.path.isfile(image_path) ):
                if class_name not in class_list:
                    train_dir = 'data/images/train/' + str(class_name)
                    val_dir = 'data/images/val/' + str(class_name)
                    class_list.add(class_name)
                    Path(train_dir).mkdir(parents=True, exist_ok=True)
                    Path(val_dir).mkdir(parents=True, exist_ok=True)

                if (split_size[class_name] == 0):
                        # copy image to val
                        copyfile(image_path, 'data/images/val/' + str(image_name))
                else:
                        # copy to train
                        copyfile(image_path, 'data/images/train/' + str(image_name))
                        split_size[class_name] -= 1



# We transform our data to fit mobilenetv2 expectations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




if __name__ == '__main__':

    preprocess_data('./data/data_set.csv')
    
    data_dir = 'data/images'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Get mobilenetv2 model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2()
    #model = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True)

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # Change the last layer to fit our needs
    print(len(class_names))
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # We will only optimize the last layer
    optimizer_ft = optim.SGD(model.classifier[1].parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=10)
    
    torch.save(model.state_dict(), "mymodel.pth")
