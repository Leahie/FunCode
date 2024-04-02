import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import sys
from tqdm.auto import tqdm



train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(),  # turns to tensor
    download = True,            
)

test_data = datasets.FashionMNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

print(train_data, test_data)

dl_train = DataLoader(train_data, batch_size=100, shuffle=True)
dl_test = DataLoader(test_data, batch_size=100, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            # First conv layer 
            nn.Conv2d( 
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),  # ReLU layer                  
            nn.MaxPool2d(kernel_size=2),     # Maxpool
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     # Now has 16 in channels bc prev layer returned 16 
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x) # does the linear data on the flattened data 
        return output, x    #
    
cnn = CNN() 

if torch.cuda.is_available():
    cnn.cuda()

# Loss Function
loss_func = nn.CrossEntropyLoss()

# Optimization Function
opt = optim.Adam(cnn.parameters(), lr=0.01)
num_epochs = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_training_steps = num_epochs * len(dl_train)
progress_bar = tqdm(range(num_training_steps))

def train(num_epochs, cnn, opt, dl_train):
    cnn.train()
    for epoch in range(1, num_epochs):
        losses = []
        for D in dl_train:
            data = Variable(D[0].to(device))
            label = Variable(D[1].to(device))
            y = cnn(data)[0]
            loss = loss_func(y, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            progress_bar.update(1)
        
def test(cnn):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dl_test:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
    

train(num_epochs, cnn, opt, dl_train)