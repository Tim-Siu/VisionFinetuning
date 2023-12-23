import os

import torch

import utils.utils as utils
from configs.paths_config import data_root, save_path
from datasets.penn_fudan import PennFudanDataset
from models.PFNet import get_model_instance_segmentation
from utils.engine import evaluate, train_one_epoch
from utils.visualize import visualize_model
from models.transform import get_transform

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ", device)

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset(data_root, get_transform(train=True))
dataset_test = PennFudanDataset(data_root, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

model_save_path = os.path.join(save_path, "model.pth")

torch.save(model.state_dict(), model_save_path)

# pick one image from the test set
# image_path = "/temp/data/PennFudanPed/PNGImages/FudanPed00046.png"
image_path = os.path.join(data_root, "PNGImages00046.png")
# output_path = "~/3/plot.png"
output_path = os.path.join(save_path, "plot.png")
visualize_model(model, device, image_path, output_path)

print("That's it!")
