#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
from torchmin import minimize_constr
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
from torch.distributions.categorical import Categorical
from bregman_kmeans import *
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment

def euclidean(x,y): return torch.clamp(torch.pow(x-y,2), min=1e-12, max=1e6)
def kl_div(x,y): return torch.clamp(x*torch.log(x/y), min=1e-12, max=1e6)
def itakura_saito(x,y): return torch.clamp(x/y - torch.log(x*y) - 1, min=1e-12, max=1e6)
def relative_entropy(x,y): return torch.clamp(x*torch.log(x/y) - x + y, min=1e-12, max=1e6)
def gamma(x,y,k): return torch.clamp(k * torch.log(y/x) + x - y, min=1e-12, max=1e6)

class BregmanClustering:
	def __init__(self,norm=-1,matrix_norm=False,method='kmeans',
			data_divergence_method = 'elementwise',
			net_divergence_method = 'elementwise',
			net_divergence="euclidean",data_divergence="euclidean",
			initialization='random'):
		self.norm = norm
		self.matrix_norm = matrix_norm
		self.method = method
		self.data_divergence_name = data_divergence
		self.net_divergence_name = net_divergence
		self.data_divergence_method = data_divergence_method
		self.net_divergence_method = net_divergence_method
		
		if method == "soft" or method == 'two_phase':
			self.net_divergence_method = 'Nelementwise'
			self.data_divergence_method = 'Nelementwise'
		if self.data_divergence_method == 'elementwise': ## As in "Graph Partitioning Based on Link Distributions"
			self.phi_data = self.get_divergence_elementwise(net_divergence)
			def data_div(X,W,mu):
				return torch.sum(self.phi_data(X,W@mu),axis=1)
			self.data_divergence = data_div
		
		else:## As in "An iterative clustering algorithm for the Contextual Stochastic Block Model with optimality guarantees"
			self.phi_data = self.get_phi(data_divergence)
			def data_div(X,W,mu):
				return torch.sum(torch.multiply(W, self.pairwise_bregman(X, mu, self.phi_data)),axis=1)
			self.data_divergence = data_div 
		
		if self.net_divergence_method == 'elementwise':
			self.phi_net = self.get_divergence_elementwise(net_divergence)				
			def net_div(A,W,B): 
				net_div = self.phi_net(A,W@B@W.T) 
				return torch.sum(net_div,axis=1) + torch.sum(net_div,axis=0)
			self.net_divergence = net_div 	
		else:
			self.phi_net = self.get_phi(net_divergence)				
			def net_div(A,W,B): 
				Z = W/W.sum(dim=0)
				Pi = Z.T@A@Z
				C = A@Z
				return torch.sum(torch.multiply(W,self.pairwise_bregman(C,Pi,self.phi_net)),axis=1)
			self.net_divergence = net_div 

		if initialization == 'random':
			self.init_variables = self.init_variables_random
		else:
			self.init_variables = self.init_variables_spectral
	
	def get_phi(self,name):
		phi_dict = {
				'euclidean': [lambda theta: theta**2, lambda theta: 2*theta, lambda theta: 2*torch.eye(theta.size()[1], dtype=torch.float64)],
				'kl_div': [lambda theta: theta * torch.log(theta), lambda theta: torch.log(theta) + 1, lambda theta: torch.eye(theta.size()[1], dtype=torch.float64) * 1/theta],
				'itakura_saito': [lambda theta: -torch.log(theta) - 1, lambda theta: -1/theta, lambda theta: torch.eye(theta.size()[1]) / (theta**2)],
				'relative_entropy': [lambda theta: theta * torch.log(theta) - theta, lambda theta: torch.log(theta), lambda theta: torch.eye(theta.size()[1]) / theta],
				'gamma': [lambda theta, k: -k + k * torch.log(k/theta), lambda theta, k: -k/theta, lambda theta, k: k * torch.eye(theta.size()[1]) / (theta**2)]
	    		  }
	    	
		return 	phi_dict[name]
	
	def get_divergence_elementwise(self,name):
		phi_dict_elementwise = {
					'euclidean': euclidean,
					'kl_div': kl_div,
					'itakura_saito': itakura_saito,
					'relative_entropy': relative_entropy,
					'gamma': gamma,
					}
		return phi_dict_elementwise[name]
  			
	#x, theta are both unidimensional, since this operation is elementwise. If they were k-dimensional, replace multiply->dot
	def bregman_divergence(self,phi_list, x, theta):
	    phi = phi_list[0]
	    gradient = phi_list[1]
	    bregman_div = phi(x) - phi(theta) - torch.multiply(gradient(theta), x-theta)
	    return bregman_div
	
	#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
	def pairwise_bregman(self,X, Y, phi_list, shape=None):
		phi = phi_list[0]
		gradient = phi_list[1]
		
		if shape:
			phi_X = torch.sum(phi(X, shape),axis=1)[:, np.newaxis]
			phi_Y = torch.sum(phi(Y, shape),axis=1)[np.newaxis, :]
		else:
			phi_X = torch.sum(phi(X),axis=1)[:, np.newaxis]
			phi_Y = torch.sum(phi(Y),axis=1)[np.newaxis, :]
		
		X = X[:, np.newaxis]
		Y = Y[np.newaxis, :]
		
		if shape:
			pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y, shape), axis=-1)
		else:
			pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y), axis=-1)
		
		return torch.clamp(pairwise_distances, min=1e-12, max=1e6)

	def init_variables_random(self,A,X,N,c,dim):
		### initialize variables
		W = torch.zeros((N,c),dtype=torch.float)
		indexes = torch.randint(low=0,high=c,size=(N,))
		for index,col in enumerate(indexes):
			W[index].index_fill_(0, col, 1)
		X = torch.tensor(X, dtype=torch.float32)
		WW_inv = torch.inverse(W.T@W)
		B = torch.eye(c)
		mu = WW_inv@W.T@X
		x_0 = torch.hstack([W.flatten(),B.flatten(),mu.flatten()])
		return x_0,W,B,mu

	def get_spectral_decomposition(self,A,c):
		L = csgraph.laplacian(A)
		L = L.astype(np.float32)
		vals = vecs = 0
		vals, vecs = eigs(L, k=(c+1), which='SM', maxiter=5000)
		U = np.delete(vecs,np.argmin(vals),1)
		return U.real

	def init_variables_spectral(self,A,X,N,c,dim):
		U = self.get_spectral_decomposition(A,c)
		W = torch.nn.functional.one_hot( torch.tensor(KMeans(n_clusters=c).fit_predict(U),dtype=int) ).float()
		A = torch.tensor(A, dtype=torch.float32)
		X = torch.tensor(X, dtype=torch.float32)
		WW_inv = torch.inverse(W.T@W)
		B = WW_inv@W.T@A@W@WW_inv
		mu = WW_inv@W.T@X
		x_0 = torch.hstack([W.flatten(),B.flatten(),mu.flatten()])
		return x_0,W,B,mu
		
	def make_obj_func_kmeans(self,A,X,N,c,dim):	
		if self.matrix_norm == False:
			def obj_func(x):
				W = x[:N*c].view(N, c)
				B = x[N*c:N*c+c*c].view(c, c)
				mu = x[N*c+c*c:].view(c,dim)
				net_divergence = self.net_divergence(A,W,B)
				data_divergence = self.data_divergence(X,W,mu)
				return torch.sum(net_divergence*data_divergence)
				#K = torch.stack((net_divergence,data_divergence),axis=-1)
				#return torch.sum(torch.pow(torch.sum(torch.pow(K, self.norm), axis=1),1/self.norm))
			return obj_func
		else:
			def obj_func(x):
				W = x[:N*c].view(N, c)
				B = x[N*c:N*c+c*c].view(c, c)
				mu = x[N*c+c*c:].view(c,dim)
				net_divergence = self.net_divergence(A,W,B)
				data_divergence = self.data_divergence(X,W,mu)
				K = torch.stack((net_divergence,data_divergence),axis=-1)
				return torch.linalg.matrix_norm(K,ord=self.norm,keepdim=False)
			return obj_func
		
	def make_obj_func_hard(self,A,X):
		def obj_func(W,B,mu):
			net_divergence = self.net_divergence(A,W,B)
			data_divergence = self.data_divergence(X,W,mu)
			K = torch.stack((net_divergence,data_divergence),axis=-1)
			return K
		return obj_func
	
	def make_obj_func_soft(self,A,X):
		def obj_func(W,B,mu):
			net_divergence = self.net_divergence(A,W,B)
			data_divergence = self.data_divergence(X,W,mu)
			K = torch.stack((net_divergence,data_divergence),axis=-1)
			return K
		return obj_func
		
	#sum of W matrix row wise is one 
	def make_constraints(self,N,c,dim):
		C = torch.cat([torch.kron(torch.eye(N),torch.ones(c)),torch.zeros((N,c*c + c*dim))],axis=1)
		constr = dict(fun = lambda x: torch.norm(C@x - torch.ones(N),1), lb=0, ub=0)
		return constr

	def make_bounds(self,N,c,dim):
		W_B_bounds = [(0,1)]*(N*c + c*c)
		mu_bounds = [(-np.inf,np.inf)]*(c*dim)
		bounds = np.vstack([W_B_bounds,mu_bounds])
		bounds_dict = dict(lb=bounds[:,0],ub=bounds[:,1])
		return bounds_dict
	
	def kmeans(self,obj_func,A,X,N,c,dim,maxiter,threshold,annealing,x_0):
		bounds = self.make_bounds(N,c,dim)
		constraints = self.make_constraints(N,c,dim)
		classes_old = classes = None
		convergence_cnt = 0
		convergence_threshold = threshold
		iter = 0
		failed_to_converge = False
		while True:
			iter += 1
			try:
				res = minimize_constr(
							obj_func, x_0, 
							max_iter=maxiter,
							constr=constraints,
							bounds=bounds
						     )
			# Check for nan values
			except:
			    print("variables are NAN'd, so terminating")
			    res.x = x_0
			    failed_to_converge = True
			    break
			#update values
			if classes is not None:
				classes_old = classes
			classes = torch.argmax(res.x[:N*c].view(N,c), axis=1)
			#Check convergence
			if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
				convergence_cnt += 1
			else:
				convergence_cnt = 0
			if (convergence_cnt == convergence_threshold) or (np.linalg.norm(res.lagrangian_grad) <= 1e-4):
				print("point assignments have converged")
				break
			
			x_0 = res.x
			if annealing:
				if self.norm > -1.0:
		        		self.norm -= .2
				elif self.norm > -120.0:
					self.norm *= 1.06
				obj_func = self.make_obj_func_kmeans(A,X,N,c,dim)
		W = res.x[:N*c].reshape((N, c))
		B = res.x[N*c:N*c+c*c].reshape((c, c))
		mu = res.x[N*c+c*c:].reshape((c, dim))
		return W,B,mu

	def hard(self,obj_func,A,X,N,c,dim,maxiter,threshold,W,B,mu):
		classes_old = classes = None
		convergence_cnt = 0
		convergence_threshold = threshold
		iter = 0
		failed_to_converge = False
		net_div0 = self.net_divergence(A,W,B).sum()
		data_div0 = self.data_divergence(X,W,mu).sum()
		while True:
			iter += 1
			print(iter)
			if self.data_divergence_method == 'elementwise' or self.net_divergence_method == 'elementwise':
				for i in range(N):
					W[i,:] = 0
					min_ = torch.tensor(np.inf)
					min_index = torch.tensor(0)
					for j in torch.arange(c):
						W[i].index_fill_(0,j,1)
						K = obj_func(W,B,mu)
						K[:,0] /= net_div0
						K[:,1] /= data_div0
						val = K.sum()
						if val < min_:
							min_index = j
							min_ = val
						W[i,j] = 0
					W[i].index_fill_(0,min_index,1)
			else:
				Z = W/W.sum(dim=0)
				Pi = Z.T@A@Z
				C = A@Z
				data_div = self.pairwise_bregman(X, mu, self.phi_data)
				net_div = self.pairwise_bregman(C, Pi, self.phi_net)
				indices = torch.argmin(net_div/net_div.sum(dim=1).reshape(-1,1).expand(-1,c)+data_div/data_div.sum(dim=1).reshape(-1,1).expand(-1,c),dim=1)
				#indices = torch.argmin(net_div/net_div0+data_div/data_div0,dim=1)
				W = torch.nn.functional.one_hot(indices).float()
			WW_inv = torch.pinverse(W.T@W)
			B = WW_inv@W.T@A@W@WW_inv
			mu = WW_inv@W.T@X
			#update values
			if classes is not None:
				classes_old = classes
			classes = torch.argmax(W, axis=1)
			#Check convergence
			if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
				convergence_cnt += 1
			else:
				convergence_cnt = 0
			if convergence_cnt == convergence_threshold or iter <= maxiter:
				print("point assignments have converged")
				break
		return W,B,mu

	def soft(self,obj_func,A,X,N,c,dim,maxiter,threshold,W,B,mu):
		classes_old = classes = None
		convergence_cnt = 0
		convergence_threshold = threshold
		iter = 0
		failed_to_converge = False
		Z = W/W.sum(dim=0)
		B = Z.T@A@Z
		C = A@Z
		communities_priors = torch.ones(c)/c
		while True:
			iter += 1
			data_div = self.pairwise_bregman(X, mu, self.phi_data)
			net_div = self.pairwise_bregman(C, B, self.phi_net)
			print(net_div[0,:],data_div[0,:])
			net_probs = torch.exp(-net_div)
			net_probs /= net_probs.sum(dim=1).reshape(-1,1).expand(-1,c) 
			data_probs = torch.exp(-data_div)
			data_probs /= data_probs.sum(dim=1).reshape(-1,1).expand(-1,c) 
			K = data_probs*net_probs
			K *= communities_priors 
			soft_assignments = K/K.sum(dim=1).reshape(-1,1).expand(-1,c)
			indices = torch.argmax(soft_assignments,dim=1)
			W = torch.nn.functional.one_hot(indices).float()
			Z = W/W.sum(dim=0)
			B = Z.T@A@Z
			C = A@Z
			mu = (soft_assignments.T@X)/soft_assignments.sum(dim=0).reshape(-1,1).expand(-1,mu.shape[1])

			#communities_priors = torch.mean(W,dim=0)
			#update values
			if classes is not None:
				classes_old = classes
			classes = torch.argmax(W, axis=1)
			#Check convergence
			if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
				convergence_cnt += 1
			else:
				convergence_cnt = 0
			if convergence_cnt == convergence_threshold or iter <= maxiter:
				print("point assignments have converged")
				break
		return W,B,mu
		
	def cluster(self,A,X,c,maxiter=100,threshold=5,annealing=False):
		N = A.shape[0]
		dim = X.shape[1]
		x_0,W,B,mu = self.init_variables(A,X,N,c,dim)
		A = torch.tensor(A,dtype=torch.float32)
		X = torch.tensor(X,dtype=torch.float32)
		
		if self.method == 'kmeans':
			obj_func = self.make_obj_func_kmeans(A,X,N,c,dim)
			W,B,mu = self.kmeans(obj_func,A,X,N,c,dim,maxiter,threshold,annealing,x_0)
			return W.detach().numpy(),B.detach().numpy(),mu.detach().numpy()
		
		elif self.method == 'hard':
			obj_func = self.make_obj_func_hard(A,X)
			print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			W,B,mu = self.hard(obj_func,A,X,N,c,dim,maxiter,threshold,W,B,mu)
			print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			return W.detach().numpy(),B.detach().numpy(),mu.detach().numpy()
		
		elif self.method == 'soft':
			obj_func = self.make_obj_func_soft(A,X)
			print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			W,B,mu = self.soft(obj_func,A,X,N,c,dim,maxiter,threshold,W,B,mu)
			print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			return W.detach().numpy(),B.detach().numpy(),mu.detach().numpy()
	
		elif self.method == "two_phase":
			obj_func = self.make_obj_func_soft(A,X)
			#print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			W,B,mu = self.two_phase(obj_func,A,X,N,c,dim,maxiter,threshold,W,B,mu)
			#print("net_div = ",self.net_divergence(A,W,B).sum(),"data_div = ",self.data_divergence(X,W,mu).sum())
			return W.detach().numpy(),B.detach().numpy(),mu.detach().numpy()						

