import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy

class BregmanKmeans: ##As in Clustering with Bregman Divergences
    def __init__(self,divergence_method = 'euclidean'):
        self.divergence_method = divergence_method
        self.phi_data = self.get_phi(divergence_method)
        def distance(X,centers):
            return self.pairwise_bregman(X, centers, self.phi_data)
        self.distance = distance		
    
    def get_phi(self,name):
        phi_dict = {
					'euclidean': [lambda theta: theta**2, lambda theta: 2*theta, lambda theta: 2*np.eye(theta.size()[1], dtype=np.float64)],
					'kl_div': [lambda theta: theta * np.log(theta), lambda theta: np.log(theta) + 1, lambda theta: np.eye(theta.size()[1], dtype=np.float64) * 1/theta],
					'itakura_saito': [lambda theta: -np.log(theta) - 1, lambda theta: -1/theta, lambda theta: np.eye(theta.size()[1]) / (theta**2)],
					'relative_entropy': [lambda theta: theta * np.log(theta) - theta, lambda theta: np.log(theta), lambda theta: np.eye(theta.size()[1]) / theta],
					'gamma': [lambda theta, k: -k + k * np.log(k/theta), lambda theta, k: -k/theta, lambda theta, k: k * np.eye(theta.size()[1]) / (theta**2)]
		   		   }
        return 	phi_dict[name]
		
    #x, theta are both unidimensional, since this operation is elementwise. If they were k-dimensional, replace multiply->dot
    def bregman_divergence(self,phi_list, x, theta):
        phi = phi_list[0]
        gradient = phi_list[1]
        bregman_div = phi(x) - phi(theta) - gradient(theta).dot(x-theta)
        return bregman_div
	
    #X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
    def pairwise_bregman(self,X, Y, phi_list, shape=None):
        phi = phi_list[0]
        gradient = phi_list[1]
		
        if shape:
            phi_X = np.sum(phi(X, shape),axis=1)[:, np.newaxis]
            phi_Y = np.sum(phi(Y, shape),axis=1)[np.newaxis, :]
        else:
            phi_X = np.sum(phi(X),axis=1)[:, np.newaxis]
            phi_Y = np.sum(phi(Y),axis=1)[np.newaxis, :]
			
        X = X[:, np.newaxis]
        Y = Y[np.newaxis, :]
			
        if shape:
            pairwise_distances = phi_X - phi_Y - np.sum((X - Y) * gradient(Y, shape), axis=-1)
        else:
            pairwise_distances = phi_X - phi_Y - np.sum((X - Y) * gradient(Y), axis=-1)	
        return np.clip(pairwise_distances, a_min=1e-12, a_max=1e6)

    def init_clusters(self,X, n_clusters, random_state, n_local_trials=None):
        n_samples, n_features = X.shape
        centers = np.empty((n_clusters, n_features), dtype=X.dtype)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))
        
        random_state = check_random_state(random_state)
		# Pick first center randomly and track index of point
        center_id = random_state.randint(n_samples)
        indices = np.full(n_clusters, -1, dtype=int)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
        else:
            centers[0] = X[center_id]
        indices[0] = center_id

        #Initialize list of closest distances and calculate current potential
        closest_dist_sq = self.distance(centers[0, np.newaxis], X)
        current_pot = closest_dist_sq.sum()

	    # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
			# XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
			# Compute distances to center candidates
            distance_to_candidates = self.distance(X[candidate_ids], X)

			# update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

			# Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

			# Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
            indices[c] = best_candidate
        return centers, indices
        
    def get_spectral_decomposition(self,A,c):
        L = csgraph.laplacian(A)
        L = L.astype(np.float32)
        vals = vecs = 0
        vals, vecs = eigs(L, k=(c+1), which='SM', maxiter=5000)
        U = np.delete(vecs,np.argmin(vals),1)
        return U.real

    def net_clustering(self,A,n_clusters,threshold,get_probs):
        N = A.shape[0]
        B = np.eye(n_clusters)
        #B[B==0] = 0.1
        indices = np.random.randint(low=0,high=n_clusters,size=(N,))
        W = np.zeros((N,n_clusters))
        rows = np.arange(N)
        W[rows,indices] = 1
        convergence_threshold = threshold
        convergence_cnt = 0
        classes_old = classes = None
        while True:
            for i in range(N):
                min_ = np.inf
                min_index = 0
                W[i,:] = 0
                for j in range(n_clusters):
                    W[i,j] = 1
                    val = ((A - W@B@W.T)**2).sum()
                    if val < min_:
                        min_index = j
                        min_ = val
                    W[i,j] = 0
                W[i,min_index] = 1
            WW_inv = np.linalg.pinv(W.T@W)
            B = WW_inv@W.T@A@W@WW_inv
            if classes is not None:
                classes_old = classes
            classes = np.argmax(W, axis=1)
            #Check convergence
            if classes_old is not None and classes is not None and np.array_equal(classes_old, classes):
                convergence_cnt += 1
            else:
                convergence_cnt = 0
            if convergence_cnt == convergence_threshold:
                print("point assignments have converged")
                break
        if get_probs:
            Z = W/W.sum(axis=0)
            W = np.exp(-self.distance(A@Z,B))
            sum_=W.sum(axis=1)
            sum_[sum_==0] = 1
            W /= sum_[:,np.newaxis]
        return W,B	
    
    def soft_clustering(self,X,n_clusters,threshold):
        classes_old = classes = None
        N = X.shape[0]
        rows = np.arange(N)
        mu,_ = self.init_clusters(X,n_clusters,42)
        soft_assignments = np.zeros((X.shape[0],n_clusters))
        convergence_threshold = threshold
        convergence_cnt = 0
        communities_priors = np.ones(n_clusters)/n_clusters
        iter_ = 0
        W = np.zeros((N,n_clusters))
        while True:
            ##E-step
            iter_+=1
            data_div = self.distance(X,mu)
            soft_assignments = np.exp(-data_div)*communities_priors[np.newaxis,:]
            total = soft_assignments.sum(axis=1)
            total[total == 0] = 1
            soft_assignments /= total[:,np.newaxis]
            ##M-step
            total_per_class = soft_assignments.sum(axis=0)[:,np.newaxis]
            mu = (soft_assignments.T@X)/total_per_class
            #update values
            if classes is not None:
                classes_old = classes
            classes = np.argmax(soft_assignments, axis=1)
            W[rows,:] = 0
            W[rows,classes] = 1
            communities_priors = np.mean(W,axis=0)
            #Check convergence
            if classes_old is not None and classes is not None and np.array_equal(classes_old, classes):
                convergence_cnt += 1
            else:
                convergence_cnt = 0
            if convergence_cnt == convergence_threshold or iter_>=100:
                print("point assignments have converged")
                break
        return soft_assignments, mu

    def hard_clustering(self,X,n_clusters,threshold):
        classes_old = classes = None
        classes_old = {}
        mu,_ = self.init_clusters(X,n_clusters,42)
        hard_assignments = np.zeros((X.shape[0],n_clusters))
        convergence_threshold = threshold
        convergence_cnt = 0
        #communities_priors = np.ones(n_clusters)/n_clusters
        rows = np.arange(X.shape[0])
        while True:
            ##E-step
            data_div = self.distance(X,mu)
            hard_assignments = np.zeros((X.shape[0],n_clusters))
            indexes = np.argmin(data_div,axis=1)
            hard_assignments[rows,indexes] = 1
            ##M-step
            #communities_priors = np.mean(hard_assignments,axis=0)
            #print("---",communities_priors)
            Z = hard_assignments/hard_assignments.sum(axis=0)
            mu = Z.T@X
            
            #update values
            if classes is not None:
                classes_old = classes
            classes = np.argmax(hard_assignments, axis=1)
			#Check convergence
            if classes_old is not None and classes is not None and np.array_equal(classes_old, classes):
                convergence_cnt += 1
            else:
                convergence_cnt = 0
            if convergence_cnt == convergence_threshold:
                #print("point assignments have converged")
                break
                           
        return hard_assignments, mu
    
    ## EM - Finds the centers jointly    
    def soft_joint_clustering(self,U,X,n_clusters,threshold):
        N = X.shape[0]
        classes_old = classes = None
        mu_data,_ = self.init_clusters(X,n_clusters,42)
        mu_net,_ = self.init_clusters(U,n_clusters,42)
        soft_assignments = np.zeros((X.shape[0],n_clusters))
        convergence_threshold = threshold
        convergence_cnt = 0
        communities_priors = np.ones(n_clusters)/n_clusters
        
        data_div = self.distance(X,mu_data)
        net_div = self.distance(U,mu_net)
        data_probs = np.exp(-data_div)
        data_probs /= data_probs.sum(axis=1)[:,np.newaxis]
        net_probs = np.exp(-net_div)
        net_probs /= net_probs.sum(axis=1)[:,np.newaxis]
        W_net = net_probs
        W_data = data_probs
        
        C = np.zeros((n_clusters,n_clusters))
        P = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                C[i,j] = np.sum((W_data[:,i] - W_net[:,j])**2) 
        row_ind, col_ind = linear_sum_assignment(C)
        P[row_ind,col_ind] = 1
        W_data = W_data@P
        mu_data = P.T @ mu_data
        iter=0
        while True:
            iter+=1
            ##E-step
            data_div = self.distance(X,mu_data)
            net_div = self.distance(U,mu_net)
            data_probs = np.exp(-data_div)
            data_probs /= data_probs.sum(axis=1)[:,np.newaxis]
            net_probs = np.exp(-net_div)
            net_probs /= net_probs.sum(axis=1)[:,np.newaxis]
                        
            soft_assignments = net_probs*data_probs
            soft_assignments /= soft_assignments.sum(axis=1)[:,np.newaxis]
            ##M-step
            #communities_priors = np.mean(soft_assignments,axis=0)
            mu_data = (soft_assignments.T@X)/soft_assignments.sum(axis=0)[:,np.newaxis]
            mu_net = (soft_assignments.T@U)/soft_assignments.sum(axis=0)[:,np.newaxis]
            #update values
            if classes is not None:
                classes_old = classes
            classes = np.argmax(soft_assignments, axis=1)
			#Check convergence
            if classes_old is not None and classes is not None and np.array_equal(classes_old, classes):
                convergence_cnt += 1
            else:
                convergence_cnt = 0
            if convergence_cnt == convergence_threshold:
                print("point assignments have converged")
                break
        return soft_assignments, mu_data
    
    ## Find independently and the permutation matrix
    def soft_joint_clustering_agreement(self,U,X,n_clusters,threshold):
        N = X.shape[0]
        net_probs,_ = self.soft_clustering(U,n_clusters,5)
        data_probs,_ = self.soft_clustering(X,n_clusters,5)
        C = np.zeros((n_clusters,n_clusters))
        P = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                C[i,j] = np.sum((data_probs[:,i] - net_probs[:,j])**2) 
        row_ind, col_ind = linear_sum_assignment(C)
        P[row_ind,col_ind] = 1
        data_probs = data_probs@P
        
        soft_assignments = data_probs.copy()
        for i in range(N):
            net_entropy = entropy(net_probs[i,:])
            data_entropy = entropy(data_probs[i,:])
            if net_entropy <= data_entropy:
                soft_assignments[i,:] = net_probs[i,:]
            else:
                soft_assignments[i,:] = data_probs[i,:]
        
        #soft_assignments = net_probs*data_probs
        #soft_assignments /= soft_assignments.sum(axis=1)[:,np.newaxis]
        return soft_assignments,None
    
    
    def soft_joint_clustering_2_step(self,U,X,n_clusters,threshold):
        N = X.shape[0]
        #net_probs,_ = self.soft_clustering(U,n_clusters,threshold)
        net_probs,_ = self.net_clustering(U,n_clusters,threshold,get_probs=False)
        data_probs,_ = self.soft_clustering(X,n_clusters,threshold)
        #A=cosine_similarity(np.hstack([net_probs,data_probs]))
        #soft_assignments,_ = self.net_clustering(A,n_clusters,threshold,get_probs=True)
        probs = np.hstack([net_probs,data_probs])
        return net_probs,_
        #soft_assignments,_ = self.hard_clustering(probs,n_clusters,threshold)
        #return np.hstack([net_probs,data_probs]),None
        #return soft_assignments,None
