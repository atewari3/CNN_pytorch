import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyper-praameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# make the images to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,
                                             download=True,transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,
                                            download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

classes = ('plane','car','bird','cat','deer'
           ,'dog','frog','horse','ship','truck')


# The function is just to see some images so you can atleast get a sense of what you are classifying
# def imshow(img):
#     img = img /(2+0.5)
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()

# data_iter = iter(train_loader)
# images,labels = next(data_iter)
# imshow(torchvision.utils.make_grid(images))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) # RGB into 6 outputs and kernel of 5
        self.pool = nn.MaxPool2d(2,2) # break into 2X2 chunks
        self.conv2 = nn.Conv2d(6,16,5) # 6inputs to 16 outputs and again kernel of 5
        self.fc1 = nn.Linear(16 * 5 * 5,120) # fully connected layer 1 input of 16*5*5 and output of 120
        self.fc2 = nn.Linear(120,84) # fully connected layer 2 input of 120 and output of 84
        self.fc3 = nn.Linear(84,10) # fully connected layer 3 input of 84 and output of 10 because we are trying to precict 10 classes



    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) # this is our first convolution but we are also appplying a relu to it and then pooling it
        x = self.pool(F.relu(self.conv2(x))) # same thing again but now with conv2
        x = x.view(-1,16*5*5) #this is essentially reshaping it so that it can however many rows/samples but we want it to be a flattend tensor
        x = F.relu(self.fc1(x)) # this is now why we are using a linear neural neural network and also applying relu
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # no need to do anything else because it will give us 10 outputs and we don't even have to do softmax because the softmax is actually
        # taken care of by Cross Entropy Loss in pyTorch
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) #Adam actually leads to a higher accuracy than SGD

n_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_steps}], Loss: {loss.item()}')

print("done with training")

PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)


# testing
with torch.no_grad():
    n_correct = 0 
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)
        
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()


        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (pred == label):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100 * n_class_correct[i]/n_class_samples[i] #we are goinf label by laberl(10 of them) and seeing how many are correct because we are doing n_class specifically
        print(f'Accuracy of {classes[i]}: {acc} %')