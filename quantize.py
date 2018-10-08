import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class QuantizedClassifier(nn.Module):
    def __init__(self, model, centers, temp=0.1, dropout=0.3):
        super(QuantizedClassifier, self).__init__()
        self.centers = centers
        self.temp = temp
        self.model = model

    def forward(self, x):
        qx = quantize(x, self.centers, self.temp)
        return self.model(qx)

class Argmin(Function):
    @staticmethod
    def forward(ctx, dists):
        '''
        x is shape (N, C)
        '''
        out = torch.zeros(dists.size(), dtype=dists.dtype)
        mins = torch.argmin(dists, dim=1).unsqueeze(1)
        out.scatter_(1, mins, 1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        grad_output is shape (N, C)
        '''
        N = grad_output.size(0)
        C = grad_output.size(1)
        out = ctx.saved_tensors[0]
        grad_input = torch.zeros_like(grad_output)
        sm_tensor = torch.zeros((N, C, C), dtype=grad_output.dtype)
        sm_tensor += out.unsqueeze(1)
        sm_tensor += torch.eye(C, dtype=grad_output.dtype)
        sm_tensor *= out.unsqueeze(2)
        grad_input = sm_tensor.permute(1, 0, 2) * grad_output
        grad_input = torch.sum(grad_input.permute(1, 0, 2), dim=2)
        return grad_input

class Softmin(Function):
    @staticmethod
    def forward(ctx, x):
        out = F.softmin(x, dim=1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        grad_output is shape (N, C)
        '''
        N = grad_output.size(0)
        C = grad_output.size(1)
        out = ctx.saved_tensors[0]
        grad_input = torch.zeros_like(grad_output)
        sm_tensor = torch.zeros((N, C, C), dtype=grad_output.dtype)
        sm_tensor += out.unsqueeze(1)
        sm_tensor += torch.eye(C, dtype=grad_output.dtype)
        sm_tensor *= out.unsqueeze(2)
        grad_input = sm_tensor.permute(1, 0, 2) * grad_output
        grad_input = torch.sum(grad_input.permute(1, 0, 2), dim=2)
        return grad_input

argmin = Argmin.apply
softmin = Softmin.apply

def quantize(x, centers, temp):
    dists = squared_distance(x.view(-1, 1), centers) / temp
    groups = argmin(dists)
    return torch.matmul(groups, centers).view(x.size())

def uniform_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    xmax = np.max(x)
    xmin = np.min(x)
    centers = (xmax - xmin)*(np.arange(levels) + 0.5)/levels + xmin
    bins = get_bins(centers)
    return centers, bins

def kmeans_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    km = KMeans(n_clusters=levels)
    km.fit(np.expand_dims(x, axis=1))
    centers = np.sort(km.cluster_centers_.reshape(-1))
    bins = get_bins(centers)
    return centers, bins

def squared_distance(x, centers):
    '''
    x has shape (N, D)
    centers has shape (C, D)
    output has shape (N, C)
    '''
    dists = torch.zeros(x.size(0), centers.size(0), dtype=x.dtype)
    for i in range(centers.size(0)):
        dists[:, i] = torch.norm(x - centers[i], dim=1)**2
    return dists
