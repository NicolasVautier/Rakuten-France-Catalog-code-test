from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import os
import sys
import getopt
from torchvision.models import mobilenet_v2
import argparse

def check_accuracy(loader, model):
    """
    loads model and compute accuracy on given dataset.
    :param loader: dataloader containing test dataset.
    :param model: model to evaluate
    :return: accuracy of input model.
    """ 
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for input, label in loader['test']:
            input = input.to(device=device)
            label = label.to(device=device)
            
            scores = model(input)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

def main(argv):
   try:
        opts, args = getopt.getopt(argv,"h:o:m:i",["modelpath=", "metric=", "imagefolder="])
   except getopt.GetoptError:
        print('evaluate.py -model <modelpath> -m <metric> -i <imagefolder>')
        sys.exit(2)
   for opt, arg in opts:
        if opt == '-h':
            print('evaluate.py -model <modelpath> -m <metric> -i <imagefolder>')
            sys.exit()
        elif opt in ("-model", "--modelpath"):
            model_pth = arg
        elif opt in ("-m", "--metric"):
            metric = arg
        elif opt in ("-i", "--imagefolder"):
            imagefolder = arg

if __name__ == '__main__':

    # Get main args
    main(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--modelpath')
    parser.add_argument('-m', '--metric')
    parser.add_argument('-i', '--imagefolder')
    args = parser.parse_args()
    model_pth = args.modelpath
    metric = args.metric
    data_dir = args.imagefolder

    supported_metris = ["accuracy"]
    if (metric not in ["accuracy"]):
        print(str(metric) + " metric currently not supported, please use one of " + str(supported_metris))
        sys.exit()

    data_dir = 'data/images'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # We transform our data to fit mobilenetv2 input expectations
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])}

    # Load images from test folder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['test']}

    class_names = image_datasets['test'].classes

    model = mobilenet_v2(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_pth))
    check_accuracy(dataloaders, model)