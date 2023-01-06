import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist

def generate_point_on_sphere(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.random.normal(loc=0,scale=1)
    x /= np.linalg.norm(x)
    return x

## Nessa versão as bases são criadas a partir de normais multivariadas

##Entrada: Matriz de probabilidades P, numero de comunidades K, um fator delta, 
## a dimensão de cada gaussiana e uma sample_along_direction que determina se a 
## amostragem do centro é ao longo de uma direção ou aleatória

##Saída: Um dicionário contendo os centros de cada gaussiana para cada P_i_j da matriz do SBM
def make_basis(P,K,delta,dim=2,sample_along_direction=True):
    basis = {}
    if sample_along_direction:
        directions = []
        for _ in range(K):
            ### amostragem de uma direção aleatoria
            directions.append(generate_point_on_sphere(dim))
        for i in range(K):
            basis[i] = {}
            for j in range(i,K):
                ### a escalagem do vetor depende da afinidade com a classe
                basis[i][j] = directions[j]*delta*(1-P[i][j])
    else:
        for i in range(K):
            basis[i] = {}
            for j in range(i,K):
                ### amostragem aleatoria do centro a partir de uma normal de media 0 sigma (1-P[i][j])*delta
                basis[i][j] = np.random.multivariate_normal(np.array([0]*dim),np.diag([delta*(1-P[i][j])]*dim))
    return basis

##Entrada: Matriz de probabilidades P, numero de comunidades K, um fator delta, 
## a dimensão de cada gaussiana, a quantidade n de vértices em cada comunidade e sample_along_direction que determina se a 
## amostragem do centro é ao longo de uma direção ou aleatória   
def make_vectors(P,K,delta,n,dim=2,sample_along_direction=True):
    X = np.zeros((K*n,dim*K))
    N = 2*n
    if delta == None:
    	dist = pdist(P)
    	#diff = np.amin(np.amin(P[P != np.amin(P,axis=0)].reshape(K,K-1),axis=1) - np.amin(P,axis=0))
    	diff = np.amin(dist)
    	delta = 2*1.1*np.sqrt( ( 1 + np.sqrt( 1+2*K/( N*np.log(N) ) ) )*np.log(N) )/(diff*np.sqrt(dim))
    basis = make_basis(P,K,delta,dim,sample_along_direction)
    for c in range(K): ## PARA CADA UMA DAS COMUNIDADES
        for i in range(n):## PARA CADA VÉRTICE DE UMA COMUNIDADE
            for j in range(K):## PARA CADA POSIÇÃO NO VETOR DE CARACTERÍSTICAS
                X[i+n*c][dim*j:dim*j+dim] = np.random.multivariate_normal(basis[min(c,j)][max(c,j)],np.eye(dim)) 
    return X
    
##Entrada: Matriz de probabilidades P, numero de comunidades K, um fator delta que controla a intensidade do sinal das features, 
## a dimensão de cada gaussiana, a quantidade n de vértices em cada comunidade e sample_along_direction que determina se a 
## amostragem do centro é ao longo de uma direção ou aleatória   
def generate_benchmark(P_net,K,delta,n,dim=2,sample_along_direction=True,P_data=None):
	if P_data is None:
		P_data = P_net
	X = make_vectors(P_data,K,delta,n,dim,sample_along_direction)
	G = nx.stochastic_block_model([n]*K,P_net,seed=42)
	for i in range(n*K):
		G.nodes[i]["attr"] = X[i,:]
	return G,nx.to_numpy_array(G),X
	
def generate_benchmark_unit_circle(P,K,sigma,n,dim,delta):
	basis = []
	for i in range(K):
		basis.append((delta*np.cos(2*np.pi*i/K),delta*np.sin(2*np.pi*i/K)))
	X = np.zeros((K*n,2))
	for c in range(K): ## PARA CADA UMA DAS COMUNIDADES
		for i in range(n):## PARA CADA VÉRTICE DE UMA COMUNIDADE
			X[i+n*c][:] = np.random.multivariate_normal(basis[c],np.diag([sigma]*2))
	G = nx.stochastic_block_model([n]*K,P,seed=42)
	for i in range(n*K):
		G.nodes[i]["attr"] = X[i,:]
	return G,nx.to_numpy_array(G),X	
