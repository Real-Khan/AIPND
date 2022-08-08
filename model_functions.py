import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import models

import processing_functions

from collections import OrderedDict

from workspace_utils import active_session

#User GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_built_model(learning_rate):
    # Build and train the neural network (Transfer Learning)
    model = models.vgg16(pretrained=True)
        
    print(model)

    # Freeze pretrained model parameters to avoid backpropogating through them
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Build custom classifier
    dropout = 0.5
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=dropout)),
                                            ('fc2', nn.Linear(5000, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    # Loss function (since the output is LogSoftmax, we use NLLLoss)
    criterion = nn.NLLLoss()

    # Gradient descent optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer, dropout

# Function for the validation pass
def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for ii, (images, labels) in enumerate(validateloader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss = criterion(output, labels)

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

#Early stopping function
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
early_stopping = EarlyStopping(tolerance=2, min_delta=5)
    
#Main function to train model
def train_model(model, initial_epochs, criterion, optimizer, train_loader, validate_loader, device=device):
    with active_session():
        epochs = initial_epochs
        steps = 0
        print_every = 5
        model.to(device)
        model.train()
        for e in range(epochs):

            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_loader):
                steps += 1

                inputs,labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optimizer.step()

                running_loss += train_loss.item()

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        validation_loss, accuracy = validation(model, validate_loader, criterion)
                        
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                        "Train Loss: {:.4f}".format(running_loss/print_every),
                        "Validation Loss {:.4f}".format(validation_loss/len(validate_loader)),
                        "Validation Accuracy: {:.4f}".format(accuracy/len(validate_loader)))
                    
                    # early stopping
                    early_stopping(train_loss, validation_loss)
                    if early_stopping.early_stop:
                        print("We are at epoch:", e)
                        break
                    
                    running_loss = 0
                    model.train()
        


# Accuracy testing function
def check_accuracy(model, test_loader):   
    model.eval()
    model.to(device)

    with torch.no_grad():
        accuracy = 0
        for ii, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            probabilities = torch.exp(output)        
            equality = (labels.data == probabilities.max(dim=1)[1])      
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = processing_functions.process_image(image_path)
    image = image.unsqueeze_(0).float()

    model.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    probabilities = F.softmax(output)
    result = probabilities.topk(topk)
    
    top_probabilities = result[0].detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    top_indices = result[1].detach().type(torch.FloatTensor).numpy().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

# Function for saving the model checkpoint
def save_checkpoint(model, epochs, training_dataset, dropout, learning_rate):
    model.class_to_idx = training_dataset.class_to_idx
    torch.save({
            'arch': "vgg16",
            'dropout': dropout,
            'epochs' : epochs,
            'learning_rate': learning_rate,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx},
            'checkpoint.pth')
    
# Function for loading the model checkpoint    
def load_model(path):
    checkpoint = torch.load(path)
    model = models.vgg16(pretrained=True)

    dropout = checkpoint['dropout']

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=dropout)),
                                        ('fc2', nn.Linear(5000, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model