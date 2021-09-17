# Rakuten-France-Catalog-code-test

Hello and welcome to my code test submission !

## Dataset

You will find a data folder containing a dataset file and the images.
This dataset is composed of product images and their associated category (product type code).
The images are present in the images folder, and the `product type codes` are present in the data_set.csv file

## Task

The goal of this test is to provide an http API to a simple mobilenet V2 classifier.
Write a simple code for training a mobilenet V2 to classify images into product categories.

Three scripts are expected:
- `train.py`: fits the model to data using an adequate loss. 
- `evaluate.py`: test the model according to an appropriate metric (for example: accuracy)
- `server.py`: runs a flask server to which a POST request containing the image can be sent and which (the server) returns the  `product_type_code` for the image.

## Tools

The code is written in Python 3 and provided with a requirements.txt file for you to install the environment.

## How to use

To start training using the custom dataset provided use : 
```console
python3 train.py -ne <num_epochs> -d <datasetpath> -i <imagefolder> -o <output> 
```

Example : 
```console
python3 train.py -ne 5 -d './data/data_set.csv' -i './data/images' -o mymodel.pth
```


