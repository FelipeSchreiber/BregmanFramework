import numpy as np

### As in https://github.com/avellal14/bregman_power_kmeans
'''
this function is structured weirdly: first 2 entries (phi, gradient of phi) can handle n x m theta matrix
last entry, only designed to work in iterative bregman update function, only works with 1 x m theta matrix and thus returns an m x m hessian
'''
def get_phi(name,elementwise=False):
    phi_dict = {
        'euclidean': [lambda theta: np.sum(theta**2, axis=1), lambda theta: 2*theta, lambda theta: 2*np.eye(theta.size()[1], dtype=np.float64)],
        'kl_div': [lambda theta: np.sum(theta * np.log(theta), axis=1), lambda theta: np.log(theta) + 1, lambda theta: np.eye(theta.size()[1], dtype=np.float64) * 1/theta],
        'itakura_saito': [lambda theta: np.sum(-np.log(theta) - 1, axis=1), lambda theta: -1/theta, lambda theta: np.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: np.sum(theta * np.log(theta) - theta, axis=1), lambda theta: np.log(theta), lambda theta: np.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: np.sum(-k + k * np.log(k/theta), axis=1), lambda theta, k: -k/theta, lambda theta, k: k * np.eye(theta.size()[1]) / (theta**2)]
    }
    phi_dict_elementwise = {
        'euclidean': [lambda theta: np.sum(theta**2), lambda theta: 2*theta, lambda theta: 2*np.eye(theta.size()[1], dtype=np.float64)],
        'kl_div': [lambda theta: np.sum(theta * np.log(theta)), lambda theta: np.log(theta) + 1, lambda theta: np.eye(theta.size()[1], dtype=np.float64) * 1/theta],
        'itakura_saito': [lambda theta: np.sum(-np.log(theta) - 1), lambda theta: -1/theta, lambda theta: np.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: np.sum(theta * np.log(theta) - theta, axis=1), lambda theta: np.log(theta), lambda theta: np.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: np.sum(-k + k * np.log(k/theta)), lambda theta, k: -k/theta, lambda theta, k: k * np.eye(theta.size()[1]) / (theta**2)]
    }
    if(elementwise):
        return phi_dict_elementwise[name]
    return phi_dict[name]


def get_bregman_divergence(phi_list):
    #x, theta are both k-dimensional
    def bregman_divergence(x, theta):
        phi = phi_list[0]
        gradient = phi_list[1]
        bregman_div = phi(x) - phi(theta) - np.dot(gradient(theta), x-theta)
        return bregman_div
    return bregman_divergence


#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
def pairwise_data_bregman(X, Y, phi_list, shape=None):
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
        pairwise_distances = phi_X - phi_Y - np.sum((X - Y) * gradient(Y, shape), axis=-1)
    else:
        pairwise_distances = phi_X - phi_Y - np.sum((X - Y) * gradient(Y), axis=-1)

    return np.clip(pairwise_distances, a_min=1e-12, a_max=1e6)

#X is n x m, y is n x m, output is n x m containing all the pairwise bregman divergences
def pairwise_net_bregman(X, Y, phi_list, shape=None):
    bregman_div = get_bregman_divergence(phi_list)
    vb = np.vectorize(bregman_div)
    return np.array(list(map(vb,X,Y)))
    