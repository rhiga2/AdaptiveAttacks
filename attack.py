import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from mnist_classifier import *

class IterativeAttack():
    def __init__(self, model, alpha=5e-3, epsilon=0.05):
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model

    def generate(self, examples, targets, num_steps=100):
        adv_examples = examples.clone()
        for i in range(num_steps):
            adv_examples.requires_grad_()
            self.model.zero_grad()
            estimates = self.model(adv_examples)
            loss = F.cross_entropy(estimates, targets)
            loss.backward()
            with torch.no_grad():
                new_examples = adv_examples + self.alpha * torch.sign(adv_examples.grad)
                total_perturb = new_examples - examples
                total_perturb = torch.clamp(total_perturb, -self.epsilon, self.epsilon)
                adv_examples = examples + total_perturb
                adv_examples = torch.clamp(adv_examples, -1, 1)
        return adv_examples

def main():
    parser = argparse.ArgumentParser(description='Iterative Method Attack')
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    args = parser.parse_args()

    model = MnistClassifier()
    model.load_state_dict(torch.load('mnist.model'))
    centers = 2*torch.arange(64, dtype=torch.float)*1/64 - 1
    qmodel = QuantizedClassifier(model, centers)
    attack = IterativeAttack(model)
    dataset = datasets.MNIST('~/Data', train=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batchsize)

    adv_data = []
    clean_data = []
    targets = []
    # Generating adversarial examples
    print('Generating Adversarial Examples')
    for idx, (data, labels) in enumerate(loader):
        if idx > 33:
            break
        clean_data.append(data)
        adv_batch = attack.generate(data, labels)
        adv_data.append(adv_batch)
        targets.append(labels)
    clean_data = torch.cat(clean_data, dim=0)
    adv_data = torch.cat(adv_data, dim=0)
    targets = torch.cat(targets)
    clean_dataset = torch.utils.data.TensorDataset(clean_data, targets)
    adv_dataset = torch.utils.data.TensorDataset(adv_data, targets)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batchsize)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batchsize)

    # Get accuracy of adversarial examples
    print('Evaluating Adversarial Examples')
    adv_correct = 0
    qadv_correct = 0
    for (data, labels) in adv_loader:
        estimate = model(data)
        adv_correct += torch.sum(torch.argmax(estimate, dim=1) == labels)
        estimate = qmodel(data)
        qadv_correct += torch.sum(torch.argmax(estimate, dim=1) == labels)

    clean_correct = 0
    qclean_correct = 0
    for (data, labels) in clean_loader:
        estimate = model(data)
        clean_correct += torch.sum(torch.argmax(estimate, dim=1) == labels)
        estimate = model(data)
        qclean_correct += torch.sum(torch.argmax(estimate, dim=1) == labels)

    adv_accuracy = adv_correct.to(torch.float) / adv_data.size(0)
    qadv_accuracy = qadv_correct.to(torch.float) / adv_data.size(0)
    qclean_accuracy = qclean_correct.to(torch.float) / clean_data.size(0)
    clean_accuracy = clean_correct.to(torch.float) / clean_data.size(0)
    print('Clean Accuracy: ', clean_accuracy)
    print('Adversarial Accuracy: ', adv_accuracy)
    print('Quantized Clean Accuracy: ', qclean_accuracy)
    print('Quantized Adversarial Accuracy: ', qadv_accuracy)

    # Attack on QuantizedClassifier
    print('Attacking Quantizer')
    qadv_data = []
    qattack = IterativeAttack(qmodel, centers)
    for (data, labels) in clean_loader:
        adv_batch = attack.generate(data, labels)
        qadv_data.append(adv_batch)
    qadv_data = torch.cat(qadv_data, dim=0)
    qadv_dataset = torch.utils.data.TensorDataset(qadv_data, targets)
    qadv_loader = DataLoader(qadv_dataset, batch_size=args.batchsize)

    qadv_correct = 0
    for (data, labels) in qadv_loader:
        estimate = qmodel(data)
        qadv_correct += torch.sum(torch.argmax(estimate, dim=1) == labels)
    qadv_accuracy = qadv_correct.to(torch.float) / qadv_data.size(0)
    print('New Attack on Quantized Model: ', qadv_accuracy)

    plt.figure()
    plt.subplot(131)
    plt.imshow(clean_data[0].squeeze(0), cmap='binary')
    plt.subplot(132)
    plt.imshow(adv_data[0].squeeze(0), cmap='binary')
    plt.subplot(133)
    plt.imshow(qadv_data[0].squeeze(0), cmap='binary')
    plt.show()


if __name__ == '__main__':
    main()
