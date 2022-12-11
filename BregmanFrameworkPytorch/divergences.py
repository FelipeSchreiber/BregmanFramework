import torch
import numpy as np 

### As in https://github.com/avellal14/bregman_power_kmeans

def dist_to_phi(dist):
    dist_to_phi_dict = {
        'gaussian': 'euclidean',
        'multinomial': 'kl_div',
        'exponential': 'itakura_saito',
        'poisson': 'relative_entropy',
        'gamma': 'gamma'
    }
    return dist_to_phi_dict[dist]

def euclidean(x,y): return torch.pow(x-y,2)
def kl_div(x,y): return x*torch.log(x/y)
def itakura_saito(x,y): return x/y - torch.log(x*y) - 1
def relative_entropy(x,y): return x*torch.log(x/y) - x + y
def gamma(x,y,k): return k * torch.log(y/x) + x - y
'''
this function is structured weirdly: first 2 entries (phi, gradient of phi) can handle n x m theta matrix
last entry, only designed to work in iterative bregman update function, only works with 1 x m theta matrix and thus returns an m x m hessian
'''
def get_phi(name,elementwise=False):
    phi_dict = {
        'euclidean': [lambda theta: torch.sum(theta**2, axis=1), lambda theta: 2*theta, lambda theta: 2*torch.eye(theta.size()[1], dtype=torch.float64)],
        'kl_div': [lambda theta: torch.sum(theta * torch.log(theta), axis=1), lambda theta: torch.log(theta) + 1, lambda theta: torch.eye(theta.size()[1], dtype=torch.float64) * 1/theta],
        'itakura_saito': [lambda theta: torch.sum(-torch.log(theta) - 1, axis=1), lambda theta: -1/theta, lambda theta: torch.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: torch.sum(theta * torch.log(theta) - theta, axis=1), lambda theta: torch.log(theta), lambda theta: torch.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: torch.sum(-k + k * torch.log(k/theta), axis=1), lambda theta, k: -k/theta, lambda theta, k: k * torch.eye(theta.size()[1]) / (theta**2)]
    }
    phi_dict_elementwise = {
        'euclidean': [euclidean],
        'kl_div': [kl_div],
        'itakura_saito': [itakura_saito],
        'relative_entropy': [relative_entropy],
        'gamma': [gamma],
    }
    if(elementwise):
        return phi_dict_elementwise[name]
    return phi_dict[name]


#x, theta are both k-dimensional
def bregman_divergence(phi_list, x, theta):
    phi = phi_list[0]
    gradient = phi_list[1]
    bregman_div = phi(x) - phi(theta) - torch.dot(gradient(theta), x-theta)
    return bregman_div

#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
def pairwise_bregman(X, Y, phi_list, shape=None):
    phi = phi_list[0]
    gradient = phi_list[1]

    if shape:
        phi_X = phi(X, shape)[:, np.newaxis]
        phi_Y = phi(Y, shape)[np.newaxis, :]
    else:
        phi_X = phi(X)[:, np.newaxis]
        phi_Y = phi(Y)[np.newaxis, :]

    X = X[:, np.newaxis]
    Y = Y[np.newaxis, :]

    if shape:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y, shape), axis=-1)
    else:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y), axis=-1)

    return torch.clamp(pairwise_distances, min=1e-12, max=1e6)
