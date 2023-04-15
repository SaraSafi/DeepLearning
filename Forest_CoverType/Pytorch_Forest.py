# Forest Cover Type dataset with Pytorch

# Import Libraries
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from torch.utils.data import Dataset, TensorDataset,DataLoader

# Check the CPU or GPU
import os
print(f'using torch{torch.__version__}({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "cpu"})')

n_input_dim = 54
n_hidden1 = 300  # Number of hidden nodes
n_hidden2 = 100
learning_rate = 0.01

def load_dataset():
    # Download dataset and save it as a csv file and then read it
    data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    )
    df = pd.read_csv(data_url, header=None)
    df.columns = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
     'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1',
     'Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3',
     'Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11',
     'Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19',
     'Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27',
     'Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35',
     'Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type']
    df.to_csv('Forest.csv')
    df=pd.read_csv('Forest.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    x=df[df.columns[:df.shape[1]-1]] # all columns except the last(Target=Cover_Type)
    y=df['Cover_Type'].values
    # Splite dataset into test and trian
    X_train, X_test, y_train, y_test = train_test_split(x, y , train_size = 0.50 ,random_state=42)
    return X_train, X_test, y_train, y_test

def preprocessing_data(X_train,X_test,y_train,y_test):
    # Normalize Only the first 10 columns and the rest of X_train features in binary(0,1)
    X_train=X_train[X_train.columns[:10]] # only the first ten columns need normalization
    scaler_t = StandardScaler().fit_transform(X_train.values)
    # Convert X_tarin to dataframe
    training_examples = pd.DataFrame(scaler_t, index=X_train.index, columns=X_train.columns)
    # Update the dataFrame with new normalize Values
    X_train.update(training_examples)
    # Convert DataFrame to Numpy array
    X_train=X_train.to_numpy()

    # Normalize Only the first 10 columns and the rest of X_test features in binary(0,1)
    X_test=X_test[X_test.columns[:10]]
    vscaler_V= StandardScaler().fit_transform(X_test.values) # this scaler uses std and mean of training dataset
    validation_examples = pd.DataFrame(vscaler_V, index=X_test.index, columns=X_test.columns)
    X_test.update(validation_examples)
    X_test=X_test.to_numpy()

    # Convert numpy Array to torch
    X_train= torch.from_numpy(X_train.astype(np.float32))
    X_test= torch.from_numpy(X_test.astype(np.float32))
    
    # Change Target type to Categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # Convert Target to torch
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test =torch.from_numpy(y_test.astype(np.float32))
    
    
    # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    X_train = TensorDataset(X_train, y_train)
    # Pytorch’s DataLoader is responsible for managing batches
    X_train = DataLoader(X_train, batch_size=64)
    
    # X_test
    X_test = TensorDataset(X_test, y_test)
    X_test = DataLoader(X_test, batch_size=32)
    
    return X_train,X_test,y_train,y_test


# A class for torch model
class TorchModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) # layer1 with 300 hidden_layers
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2) # layer2 with 100 hidden Layesr
        self.layer_out = nn.Linear(n_hidden2, 8) # The number of input=8
        
        
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
   

def final():
    #call load_dataset 
    x_train, x_val, y_train, y_val= load_dataset()
    # print('x',x_train.shape,'y',y_train.shape)
    # print('x',x_val.shape,'y',y_val.shape)
    
    # prepare data
    x_train,x_val,y_train,y_val= preprocessing_data(x_train,x_val,y_train,y_val)
    
    # call Model Calss
    model = TorchModel()
    print(model)
    
    # Define loss_func and Optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 10
    model.train()
    train_loss = []
    for epoch in range(epochs):
        #Within each epoch run the subsets of data = batch sizes.
        for xb, yb in x_train:
            y_pred = model(xb)            # Forward Propagation
            loss = loss_func(y_pred, yb)  # Loss Computation
            optimizer.zero_grad()         # Clearing all previous gradients, setting to zero 
            loss.backward()               # Back Propagation
            optimizer.step()   # Updating the parameters 
            acc = (y_pred.round() == yb).float().mean()
        print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
        print("accuracy in iteration :"+str(epoch)+" is: "+str(acc.item()))
        train_loss.append(loss.item())
    print('Last iteration loss value: '+str(loss.item()))
final()


'''The Result:
Loss in iteration :0 is: 1.5873680114746094
accuracy in iteration :0 is: 0.862500011920929
Loss in iteration :1 is: 1.4360363483428955
accuracy in iteration :1 is: 0.875
Loss in iteration :2 is: 1.5041040182113647
accuracy in iteration :2 is: 0.875
Loss in iteration :3 is: 1.4990227222442627
accuracy in iteration :3 is: 0.875
Loss in iteration :4 is: 1.418455719947815
accuracy in iteration :4 is: 0.887499988079071
Loss in iteration :5 is: 1.485687255859375
accuracy in iteration :5 is: 0.887499988079071
Loss in iteration :6 is: 1.5029575824737549
accuracy in iteration :6 is: 0.875
Loss in iteration :7 is: 1.4201934337615967
accuracy in iteration :7 is: 0.887499988079071
Loss in iteration :8 is: 1.4205348491668701
accuracy in iteration :8 is: 0.887499988079071
Loss in iteration :9 is: 1.4042541980743408
accuracy in iteration :9 is: 0.8999999761581421
Last iteration loss value: 1.4042541980743408'''