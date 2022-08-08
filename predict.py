import model_functions as mf
import processing_functions as pf
import argparse

parser = argparse.ArgumentParser(description='Image Classifier Model Predictions')

# Command line arguments
parser.add_argument('--image_path', type = str, default = 'flowers/test/15/image_06351.jpg', 
                    help = 'Path to image')
parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint.pth',
                    help = 'Path to checkpoint file')
parser.add_argument('--topk', type = int, default = 5, 
                    help = 'Top k classes and probabilities')
parser.add_argument('--file', type = str, default = 'cat_to_name.json', 
                    help = 'labels json file')

arguments = parser.parse_args()

# Label Maping
name_dict = pf.load_json(arguments.file)

# Load pretrained model
model = mf.load_model(arguments.checkpoint_path)
print(model)

# Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
image = pf.process_image(arguments.image_path)

# Image Show
pf.imshow(image)

# Return top k probabilities and classes in prediction
probs, classes = mf.predict(arguments.image_path, model, arguments.topk)
print(probs)
print(classes)

# Sanity Checking - Show the highest top K prediction
pf.sanity_check(arguments.image_path, name_dict, classes, probs)