# Rakuten-France-Catalog-code-test

Hello and welcome to my code test submission !

## Dataset

You will find a data folder containing a dataset file and the images.
This dataset is composed of product images and their associated category (product type code).
The images are present in the images folder, and the `product type codes` are present in the data_set.csv file

## Scripts

Three scripts are provided:
- `train.py`: fits the model to data using an adequate loss. 
- `evaluate.py`: test the model according to an appropriate metric (for example: accuracy)
- `server.py`: runs a flask server to which a POST request containing the image can be sent and which (the server) returns the  `product_type_code` for the image.

## Tools

The code is written in Python 3 and provided with a requirements.txt file for you to install the environment.

## Training

First, unzip the data.zip file.

Then start training using the custom dataset provided use : 
```console
python3 train.py -ne <num_epochs> -d <datasetpath> -i <imagefolder> -o <output> 
```

Example : 
```console
python3 train.py -ne 5 -d './data/data_set.csv' -i './data/images' -o mymodel.pth
```

## Evaluate

To evaluate your model run : 

```console
python3 evaluate.py -model <modelpath> -m <metric> -i <imagefolder>
```

Example :

```console
python3 evaluate.py -model mymodel.pth -m accurac -i './data/images' 
```

## Server

Run the folowing command to start the server : 
```console
python3 server.py
```

Then in another terminal, run : 

```console
curl -F "file=@image_path" http://localhost:5000/predict
```

Example : 

```console
curl -F "file=@data/images/image_285526945_product_16273363.jpg" http://localhost:5000/predict
```
