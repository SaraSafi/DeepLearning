# Captcha with pytorch

# Import Libraries
import os
import glob 
import pandas as pd
import string
import collections
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim


# Path of dataset
path = './a/data/'
# The function for creating the dataframe from dataset
def Load_dataset():
    # Open the dataset
    data = glob.glob(os.path.join('./a/data/', '*.png'))
    # try to encoding labels with ascii letters and save them in a Dataframe
    all_letters = string.ascii_uppercase + string.digits+string.ascii_lowercase
    mapping={}
    mapping_inv = {}
    i = 1
    for x in all_letters:
        mapping[x] = i
        mapping_inv[i] = x
        i += 1
    # The number of class
    num_class = len(mapping)
    print(num_class)
    # make a dataset
    images = [] # list for saving images
    labels = [] # list for saving labels
    # create a dictionary
    datas = collections.defaultdict(list)
    for d in data:
        x = d.split('/')[-1]
        datas['image'].append(x)
        datas['label'].append([mapping[i] for i in x.split('.')[0]])
    # Save dictionary to DataFrame
    df = pd.DataFrame(datas)
    
    return df

'''
	image 	label
0 	3scuV.png 	[30, 55, 39, 57, 22]'''


# create a captchadatset 
class CaptchaDataset:
    def __init__(self, df, transform=None):
        self.df = df
        # torchvision.transforms to Composes several transforms together
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # Use PIL Library: ‘L’ convert function converts the image from it’s regular RGB colors to simple black and white (gray-scale).
        image = Image.open(os.path.join(path, data['image'])).convert('L')
        # Convert labels to torch
        label = torch.tensor(data['label'], dtype=torch.int32)
        # print(image)
        # print(label)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

 

# Create Bidirectional 
class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)     
        return out


# CRNN model for Captcha
class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256))
        
        self.linear = nn.Linear(5888, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None):
        out = self.cnn(X)
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)
        out = self.linear(out)

        out = out.permute(1, 0, 2)
        out = self.rnn(out)
            
        if y is not None:
            T = out.size(0)
            N = out.size(1)
        
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
        
            loss = criterion(out, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None
    
    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)


   

# Train the model
def final():
    # Load dataset
    df=Load_dataset()
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    # Composes several transforms together
    transform = T.Compose([
    T.ToTensor()
    ])

    # Load data as CaptchaDataset
    train_data = CaptchaDataset(df_train, transform)
    test_data = CaptchaDataset(df_test, transform)

    # Pytorch’s DataLoader is responsible for managing batches
    # And use train_loader in each epochs
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    # Create Device to check CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # The output of model is number of classes that is equal to 62
    model = CRNN(in_channels=1, output=62).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # CTCLoss:Calculates loss between a continuous (unsegmented) time series and a target sequence
    criterion = nn.CTCLoss()
    

    # Within each epoch run the subsets of data = batch sizes
    hist_loss = []
    for epoch in range(2):
            model.train()
            # tqdm is used to create a smart progress bar for the loops
            tk = tqdm(train_loader, total=len(train_loader))
            for data, target in tk:
                data = data.to(device=device)
                target = target.to(device=device)

                optimizer.zero_grad() # Clearing all previous gradients, setting to zero 
                out, loss = model(data, target, criterion=criterion) ## Loss Computation
                loss.backward() # Back Propagation
                optimizer.step() # Updating the parameters
                tk.set_postfix({'Epoch':epoch+1, 'Loss' : loss.item()})
    print('Last iteration loss value: '+str(loss.item()))
final()   

'''The final result with 100 epochs: 
100%|██████████| 54/54 [00:07<00:00,  7.38it/s, Epoch=98, Loss=0.112]
100%|██████████| 54/54 [00:07<00:00,  7.49it/s, Epoch=99, Loss=0.144]
100%|██████████| 54/54 [00:07<00:00,  7.47it/s, Epoch=100, Loss=0.139]

Last iteration loss value: 0.13945823907852173
'''