# Import libraries
from PIL import Image
import torchvision
import numpy as np
import shutil
import random
import torch
import os
import random

# The direction of images
train_dirs = {
    'normal': 'Xray/data/normal',
    'opacity': 'Xray/data/opacity',}
test_dirs = {
    'normal': 'Xray/test_data/normal',
    'opacity': 'Xray/test_data/opacity',}


class Create_XRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name): # define images in each classes
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('jpeg')] 
            return images
        
        self.images = {}
        self.class_names = ['normal', 'opacity']
        
        for c in self.class_names:
            self.images[c] = get_images(c)
            
        self.image_dirs = image_dirs
        self.transform = transform
    # len(dataset) returns the size of the dataset.     
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index%len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

def train_test():
    # Use torchvision transform to prepare images
    train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = Create_XRayDataset(train_dirs, train_transform)
    test_dataset = Create_XRayDataset(test_dirs, test_transform)
    
    return train_dataset,test_dataset

def final(epochs):
    # Create train and test dataset
    train_dataset,test_dataset=train_test()
    batch_size = 6
    # use pytorch dataloader ,
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    # Use pretrrained model
    resnet18 = torchvision.models.resnet18(pretrained = True)
    resnet18.fc = torch.nn.Linear(in_features = 512, out_features = 2)
    loss_fn     = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(resnet18.parameters(), lr = 3e-5)
    
    
    for e in range(0, epochs):
        
        print(f'Starting epoch {e + 1}/{epochs}')

        train_loss = 0.
        val_loss = 0.

        resnet18.train() # set model to training phase
        # calculate the validation loss for each step in each epochs
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 10 == 0:
                print('Evaluating at step', train_step)

                accuracy = 0

                resnet18.eval() # set model to eval phase
                

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


                resnet18.train()
                if accuracy >= 0.90:
                    
                    print('The best accuracy:',accuracy)
                    break
                

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
final(epochs=1)


    
