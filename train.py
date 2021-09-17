from __future__ import print_function, division
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import csv
import getopt
import sys
from pathlib import Path
import argparse
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
        # We skip the first line as it contains header
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
        n_image_per_classes[key] = round(n_image_per_classes[key] * 0.9)
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
            # For each class we create a folder that will contain our train/test images
            if (os.path.isfile(image_path) ):
                if class_name not in class_list:
                    train_dir = 'data/images/train/' + str(class_name)
                    test_dir = 'data/images/test/' + str(class_name)
                    class_list.add(class_name)
                    Path(train_dir).mkdir(parents=True, exist_ok=True)
                    Path(test_dir).mkdir(parents=True, exist_ok=True)

                if (split_size[class_name] == 0):
                        # copy image to test folder
                        copyfile(image_path, 'data/images/test/' + str(image_name))
                else:
                        # copy to train folder
                        copyfile(image_path, 'data/images/train/' + str(image_name))
                        split_size[class_name] -= 1




def train_model(model, criterion, optimizer, num_epochs):
    """
    loads model and process to training.
    :param dataset: csv file containing images names and associated labels.
    :return: nothing.
    """ 
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # forward
            # track history
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes["train"]

        print('{} Loss: {:.4f}'.format(
            "train", epoch_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load last model weights
    model.load_state_dict(model)
    return model


def main(argv):
   try:
        opts, args = getopt.getopt(argv,"h:ne:o:i:d",["num_epochs=", "output=", "datasetpath=", "imagefolder="])
   except getopt.GetoptError:
        print('train.py -ne <num_epochs> -o <output> -d <datasetpath> -i <imagefolder>')
        sys.exit(2)
   for opt, arg in opts:
        if opt == '-h':
            print('train.py -ne <num_epochs> -o <output> -d <datasetpath> -i <imagefolder>')
            sys.exit()
        elif opt in ("-ne", "--num_epochs"):
            epoch = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
        elif opt in ("-d", "--datasetpath"):
            datasetpath = arg
        elif opt in ("-i", "--imagefolder"):
            imagefolder = arg


if __name__ == '__main__':

    main(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('-ne', '--epoch')
    parser.add_argument('-d', '--datasetpath')
    parser.add_argument('-i', '--imagefolder')
    args = parser.parse_args()
    model_pth = args.output
    num_epochs = int(args.epoch)
    dataset = args.datasetpath
    data_dir = args.imagefolder

    preprocess_data(dataset)

    # We transform our data to fit mobilenetv2 input expectations
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])}

    # Load images from train folder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get mobilenetv2 model
    model = mobilenet_v2()

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # Change the last layer to fit our needs
    print(len(class_names))
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # We will only optimize the last layer
    optimizer_ft = optim.Adam(model.classifier[1].parameters(), 0.01)

    model = train_model(model, criterion, optimizer_ft, num_epochs)
    
    torch.save(model.state_dict(), model_pth)
