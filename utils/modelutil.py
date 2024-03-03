import torch
import torch.nn as nn
from torchvision import transforms, datasets

from utils.flutil import Flutils as flutils

# Define the model
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(7*7*32, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ModelUtils:
     
     def __init__(self) -> None:
          pass
     
     @staticmethod
     def load_data(batch_size=32):
        """
        load FashionMNIST data
        Args:
            batch_size (int, optional): the batch size. Defaults to 32.
        """        
        batch_size = batch_size
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ),)]) # normalize to [-1,1]
        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        
        classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return trainset, testset, trainloader, testloader
    
     
     @staticmethod
     def test_total_accuracy(model, testloader, device="cpu"):
        if type(device) == str:  
            device = torch.device(device)
        model = model.to(device) 
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
     
     # You will have to implement the attacks here.
     @staticmethod
     def modelTrainer(model, num_epochs, trainloader, optimizer, criterion, device="cpu",attack=False):
        if type(device) == str:  
            device = torch.device(device) 
        model = model.to(device)
        
        model.train()

        train_losses = []
        for _ in range(num_epochs):  # loop over the dataset multiple time
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)# get the inputs; data is a list of [inputs, labels]
                if attack:
                    inputs, labels = flutils.randomFlipAttack(inputs, labels)
                    labels = labels.to(device)
                inputs.to(device) 
                labels.to(device)
                optimizer.zero_grad() # zero the parameter gradients

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                #if i % 50 == 0 and i != 0:  
                #    print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / 50, 3)}")
                #    running_loss = 0.0
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(trainloader)) #this is buggy
        return train_losses


