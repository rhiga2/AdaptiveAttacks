import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from quantize import *

class MnistClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(320, 50)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()
    if verbose:
        print('Epoch %d: Train Loss %f' % (epoch, total_loss / (batch_idx + 1)))

def evaluate_epoch(model, device, val_loader, epoch, verbose=True):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total += data.size(0)
        total_loss += loss
        correct = torch.sum(torch.argmax(output, dim=1) == target)
        total_correct += correct
    print('Accuracy %f' % (100 * total_correct.to(torch.float) / total))

def main():
    parser = argparse.ArgumentParser(description='MNIST classifier')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=8)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device('cpu')
    train_data = datasets.MNIST('~/Data', train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batchsize)
    val_data = datasets.MNIST('~/Data', train=False, transform=transforms.ToTensor())
    val_loader = DataLoader(val_data, shuffle=True, batch_size=args.batchsize)
    model = MnistClassifier()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.pretrainaed:
        for epoch in range(args.epochs):
            train_epoch(model, device, train_loader, optimizer, epoch)
            evaluate_epoch(model, device, val_loader, epoch)
            torch.save(model.state_dict(), 'mnist.model')
    else:
        model.load_state_dict(torch.load('mnist.model'))

    centers = 2*torch.arange(200, dtype=torch.float)*1/200 - 1
    qmodel = QuantizedClassifier(model, centers)
    evaluate_epoch(qmodel, device, val_loader, 0)

if __name__ == '__main__':
    main()
