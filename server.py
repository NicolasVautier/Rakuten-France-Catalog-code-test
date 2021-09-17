from flask import Flask, request, render_template
import jsonpickle
import numpy as np
from PIL import Image
import torch
import csv
from torchvision.models import mobilenet_v2
from torchvision import transforms

# Initialize the Flask application
app = Flask(__name__)
dataset = './data/data_set.csv'

def get_classes(dataset):
    class_list = []
    with open(dataset, newline='') as csvfile:
        next(csvfile)
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            class_number = str(row).split(",")[1][:-2]

            if int(class_number) not in class_list:
                class_list.append(int(class_number))
    class_list.sort()
    return class_list

# route http posts to this method
@app.route('/predict', methods=['POST'])
def predict_image():
    labels = get_classes(dataset)
    image = Image.open(request.files['file'])

    # We transform our data to fit mobilenetv2 input expectations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model
    model = mobilenet_v2(num_classes=32)
    model.load_state_dict(torch.load('mymodel.pth'))
    model.eval()

    # Process image for inference
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)

    output = model(image_tensor)
    index = labels[output.data.cpu().numpy().argmax()]
    return "\n" + "Product type code is : " + str(index)


# start flask app
app.run(host="0.0.0.0", port=5000)