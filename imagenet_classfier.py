import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from quantize import *

def main():
    parser = argparse.ArgumentParser(description='Imagenet classifier')
    parser.add_argument('--datapath', default='/media/data/Imagenet')
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
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = torch.device('cuda:0')
    model = models.resnet50(pretrained=True).to(device)
    qmodel = QuantizedClassifier(model, centers=centers)
    centers = 2*torch.arange(64, dtype=torch.float)*1/64 - 1

    print('Attacking Non-Quantized Model')
    for (data, labels) in valloader:
        pass

    print('Evaluating Non-Adaptive Attack')

    print('Evaluating Adaptive Attack')

    print('Attacking Quantized Model')
    for (data, labels) in valloader:
        pass

if __name__=='__main__':
    main()
