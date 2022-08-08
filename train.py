import model_functions as mf
import processing_functions as pf
import argparse

parser = argparse.ArgumentParser(description='Image Classifier Model Training')

# Command line arguments
parser.add_argument('--learning_rate', type = float, default = 0.001, 
                    help = 'Learning Rate')
parser.add_argument('--epochs', type = int, default = 15, 
                    help = 'Epochs')

arguments = parser.parse_args()

# Image data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Load data function
training_dataset, train_loader, validate_loader, test_loader = pf.transform_load_data(train_dir, valid_dir, test_dir)

#Customize model
model, criterion, optimizer, dropout = mf.custom_built_model(arguments.learning_rate)
print("Customized model: ", model)
print("Loss Function: ", criterion)
print("Optimizer: ", optimizer)
print("Dropout: ", dropout)

#Train Mode
mf.train_model(model, arguments.epochs, criterion, optimizer, train_loader, validate_loader)

#Check for accuracy
mf.check_accuracy(model, test_loader)

#Save checkpoint
mf.save_checkpoint(model, arguments.epochs, training_dataset, dropout, arguments.learning_rate)