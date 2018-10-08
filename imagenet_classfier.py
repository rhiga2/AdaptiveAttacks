import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from quantize import *

def num_correct(logits, labels):
    return torch.sum(torch.argmax(logits, dim=1) == labels)

def main():
    parser = argparse.ArgumentParser(description='Imagenet classifier')
    parser.add_argument('--datapath', default='/media/data/Imagenet')
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    valdir = os.path.join(args.datapath, 'val')
    valloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])),
        batch_size=args.batchsize, shuffle=False, pin_memory=True)

    device = torch.device('cuda:0')
    model = models.resnet50(pretrained=True).to(device)
    qmodel = QuantizedClassifier(model, centers=centers)
    centers = 2*torch.arange(64, dtype=torch.float)*1/64 - 1

    print('Attacking Non-Quantized Model')
    correct = 0
    for batch_idx, (data, labels) in enumerate(valloader):
        if batch_idx > 1024 // args.batchsize:
            break
        logits = model(data)
        print(logits.size())
        correct += num_correct(logits, labels)
    accuracy = correct.to(torch.float) / 1024

    print('Evaluating Non-Adaptive Attack')

    print('Evaluating Adaptive Attack')

    print('Attacking Quantized Model')

if __name__=='__main__':
    main()
