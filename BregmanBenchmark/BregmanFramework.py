#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
from torchmin import minimize_constr

def euclidean(x,y): return torch.clamp(torch.pow(x-y,2), min=1e-12, max=1e6)
def kl_div(x,y): return torch.clamp(x*torch.log(x/y), min=1e-12, max=1e6)
def itakura_saito(x,y): return torch.clamp(x/y - torch.log(x*y) - 1, min=1e-12, max=1e6)
def relative_entropy(x,y): return torch.clamp(x*torch.log(x/y) - x + y, min=1e-12, max=1e6)
def gamma(x,y,k): return torch.clamp(k * torch.log(y/x) + x - y, min=1e-12, max=1e6)

class BregmanClustering:
	def __init__(self,norm=-1,matrix_norm=False,method='kmeans'):
		self.norm = norm
		self.matrix_norm = matrix_norm
		self.method = method
	
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
	
	def init_variables(self,A,X,N,c,dim):
		### initialize variables
		W = torch.zeros((N,c),dtype=torch.float)
		indexes = torch.randint(low=0,high=c,size=(N,))
		for index,col in enumerate(indexes):
        		W[index].index_fill_(0, col, 1)
		WW_inv = torch.inverse(W.T@W)
		B = WW_inv@W.T@A@W@WW_inv
		mu = WW_inv@W.T@X
		x_0 = torch.hstack([W.flatten(),B.flatten(),mu.flatten()])
		return x_0,W,B,mu
	
	def make_obj_func_kmeans(self,phi_net,phi_data,A,X,N,c,dim):	
		if self.matrix_norm == False:
			def obj_func(x):
				W = x[:N*c].view(N, c)
				B = x[N*c:N*c+c*c].view(c, c)
				mu = x[N*c+c*c:].view(c,dim)
				net_divergence = torch.sum(phi_net(A,W@B@W.T),axis=1)
				data_divergence = torch.sum(torch.multiply(W, self.pairwise_bregman(X, mu, phi_data)),axis=1)
				K = torch.stack((net_divergence,data_divergence),axis=-1)
				return torch.sum(torch.pow(torch.sum(torch.pow(K, self.norm), axis=1),1/self.norm))
			return obj_func
		else:
			def obj_func(x):
				W = x[:N*c].view(N, c)
				B = x[N*c:N*c+c*c].view(c, c)
				mu = x[N*c+c*c:].view(c,dim)
				#net_divergence = torch.sum(self.bregman_divergence(phi_net,A,W@B@W.T),axis=1)
				net_divergence = torch.sum(phi_net(A,W@B@W.T),axis=1)
				data_divergence = torch.sum(torch.multiply(W, self.pairwise_bregman(X, mu, phi_data)),axis=1)
				K = torch.stack((net_divergence,data_divergence),axis=-1)
				return torch.linalg.matrix_norm(K,ord=self.norm,keepdim=False)
			return obj_func
		
	def make_obj_func_hard(self,phi_net,phi_data,A,X,W_0,B_0,mu_0):	
		#Z = W_0/W_0.sum(dim=0)
		#net_div0 = torch.multiply(W_0, self.pairwise_bregman(A@Z,Z.T@A@Z, phi_data)).sum()
		#net_div = phi_net(A,W_0@B_0@W_0.T)
		#net_divergence = torch.sum(net_div,axis=1) + torch.sum(net_div,axis=0)
		#net_div0 = net_divergence.sum()
		#data_div0 = torch.multiply(W_0, self.pairwise_bregman(X, mu_0, phi_data)).sum()
		data_div0 = net_div0 = 1
		def obj_func(W,B,mu):
			net_div = phi_net(A,W@B@W.T)
			net_divergence = (torch.sum(net_div,axis=1) + torch.sum(net_div,axis=0))/net_div0
			#Z = W/W.sum(dim=0)
			#net_divergence = torch.sum(torch.multiply(W, self.pairwise_bregman(A@Z,Z.T@A@Z, phi_data)),axis=1)/net_div0
			data_divergence = torch.sum(torch.multiply(W, self.pairwise_bregman(X, mu, phi_data)),axis=1)/data_div0
			#print("net ",net_divergence.sum(),"div ",data_divergence.sum())
			K = torch.stack((net_divergence,data_divergence),axis=-1)
			return K.sum()
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
	
	def kmeans(self,obj_func,phi_net,phi_data,A,X,N,c,dim,maxiter,threshold):
		bounds = self.make_bounds(N,c,dim)
		constraints = self.make_constraints(N,c,dim)
		x_0,_,_,_ = self.init_variables(A,X,N,c,dim)
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
			#if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
			#	convergence_cnt += 1
			#else:
			#	convergence_cnt = 0
			#if convergence_cnt == convergence_threshold and torch.res.jac:
			#	print("point assignments have converged")
			#	break
			
			if np.linalg.norm(res.lagrangian_grad) <= 1e-4:
				print("point assignments have converged")
				break
			x_0 = res.x
			if self.norm > -1.0:
                		self.norm -= .2
			elif self.norm > -120.0:
				self.norm *= 1.06
			obj_func = self.make_obj_func_kmeans(phi_net,phi_data,A,X,N,c,dim)
		W = res.x[:N*c].reshape((N, c)).detach().numpy()
		B = res.x[N*c:N*c+c*c].reshape((c, c)).detach().numpy()
		mu = res.x[N*c+c*c:].reshape((c, dim)).detach().numpy()
		return W,B,mu
		
	def hard(self,obj_func,A,X,N,c,dim,threshold,W,B,mu):
		classes_old = classes = None
		convergence_cnt = 0
		convergence_threshold = threshold
		iter = 0
		failed_to_converge = False
		while True:
			iter += 1
			for i in range(N):
				W[i,:] = 0
				min_ = torch.tensor(np.inf)
				min_index = torch.tensor(0)
				for j in torch.arange(c):
					W[i].index_fill_(0,j,1)
					val = obj_func(W,B,mu)
					if val < min_:
						min_index = j
						min_ = val
					W[i,j] = 0
				W[i].index_fill_(0,min_index,1)
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
			if convergence_cnt == convergence_threshold:
				print("point assignments have converged")
				break
		return W.detach().numpy(),B.detach().numpy(),mu.detach().numpy()
		
	def cluster(self,A,X,c,net_divergence="euclidean",data_divergence="euclidean",maxiter=100,threshold=5):
		phi_net = self.get_divergence_elementwise(net_divergence)
		phi_data = self.get_phi(data_divergence)
		N = A.size(dim=0)
		dim = X.size(dim=1)
		if self.method == 'kmeans':
			obj_func = self.make_obj_func_kmeans(phi_net,phi_data,A,X,N,c,dim)
			return self.kmeans(obj_func,phi_net,phi_data,A,X,N,c,dim,maxiter,threshold)
		else:
			_,W,B,mu = self.init_variables(A,X,N,c,dim)
			obj_func = self.make_obj_func_hard(phi_net,phi_data,A,X,W,B,mu)
			return self.hard(obj_func,A,X,N,c,dim,threshold,W,B,mu)
