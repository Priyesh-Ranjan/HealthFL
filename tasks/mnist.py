from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from dataloader import *
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

class Net(nn.Module):
    '''
    LeNet

    retrieved from the pytorch tutorial
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

    '''

    def __init__(self, weight_scale=0.1, rho_offset=-3, zeta=10):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # The BNN part
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.]*len(self.layer_param_shapes, device=self.device))
        
        for shape in self.layer_param_shapes :
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale*torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)
            
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos) :
        return [F.softplus(rho) for rho in rhos]
    
    def transform_gaussian_samples(self, mus, rhos, epsilons) :
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)):
            samples.append(mus[j] + self.sigmas[j]*epsilons[j])
        return samples

    def samples_epsilons(self, param_shapes) :
        epsilons = [torch.normal(mean=torch.zeros(shape),
                                 std =0.001*torch.ones(shape)).to(self.device) for shape in param_shapes]
        return epsilons
    
    def log_softmax_likelihood(self, yhat_linear, y) :
        return torch.nansum(y * F.log_softmax(yhat_linear), dim=0)
    
    def combined_loss_personal(self, output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches) :
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]), Normal(mus_local[i].detach(), sigmas_local[i].detach()))) for i in range(len(params))])
        
        return 1.0 / num_batches * (self.zeta * KL_q_w) - log_likelihood_sum
    
    def combined_loss_local(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                                              Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        return 1.0 / num_batches * (self.zeta*KL_q_w)

def getDataset():
    dataset = datasets.MNIST('./data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], 'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                             transform=transforms.Compose(
                                                                 [transforms.Resize((32, 32)), transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                              batch_size=test_batch_size, shuffle=True)
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (1, 32, 32))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
