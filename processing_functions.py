import torch
from torchvision import datasets, transforms

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from PIL import Image

import json

# Loader data set function
def transform_load_data(train_dir, valid_dir, test_dir):
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

    return training_dataset, train_loader, validate_loader, test_loader

# Function for processing a PIL image for use in the PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)

    process_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = process_image(pil_image)

    return processed_image

# Function to convert a PyTorch tensor and display it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# Load class_to_name json file 
def load_json(json_file):  
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


# Function to display an image along with the top 5 classes
def sanity_check(img_path, name_dict, classes, probs):

    image = process_image(img_path)

    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)
    flower_title = name_dict[classes[0]]
    imshow(image, plot_1, title=flower_title)

    flower_names = [name_dict[i] for i in classes]
    plot_2 = plt.subplot(2,1,2)
    sb.barplot(x=probs, y=flower_names)

    plt.show()