import os
import torch
from models.PFNet import get_model_instance_segmentation
from utils.visualize import visualize_model
from configs.paths_config import data_root, save_path

# Function to load the model
def load_model(num_classes, model_path, device):
    # Load an instance segmentation model pre-trained on a custom dataset
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: ", device)

# Define the number of classes and the path to the model
num_classes = 2  # background and person
model_path = os.path.join(save_path, "model.pth")

# Load the pre-trained model
model = load_model(num_classes, model_path, device)

# Set the model to evaluation mode
model.eval()

# Define the path for the image you want to visualize
image_path = os.path.join(data_root, "PNGImages", "FudanPed00046.png")

# Define the output path for the visualization
output_path = os.path.join(save_path, "visualization.png")

# Perform the visualization
visualize_model(model, device, image_path, output_path)

print("Visualization complete! Check the output at:", output_path)
