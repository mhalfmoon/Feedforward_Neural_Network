import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784      # Image size = 28 x 28 = 784
hidden_size = 500     # The number of nodes at the hidden layer
num_classes = 10      # The number of output classes. [0-9]
num_epochs = 5        # The number of times the dataset is trained
batch_size = 100      # The size of the input data for one iteration
learning_rate = 0.001 # The speed of convergence
# Download the dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())
# Load the dataset. The training set is shuffled to train
# independently of data order.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
net = Net(input_size, hidden_size, num_classes)
# Loss function determines how the output is compared to a class.
# This determines how good the neural network performs.
# The optimizer chooses how to update the weight and convergence
# to the best weights.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Load a set of images with (index, data, class)
    for i, (images, labels) in enumerate(train_loader):
        # Change image from a vector to a matrix
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        #Initialize the hidden weight to zeros
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1) #Choose the best classe
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the 10K test images: %d %%' %(100*correct / total))

torch.save(net.state_dict(), 'fnn_model.pkl')
