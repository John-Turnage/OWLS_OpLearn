"""
Includes methods for induced sampling from various tensor product probability measures
"""
import numpy as np
from induced_dist import Jacobi_Induced_Dist

def Jacobi_Tensor_Induced_Sampling(M,PolyIdxs,JacobiParams,max_poly_degree):
    """
    Params: 
        M (int): Number of desired samples
        PolyIdxs(nmpy  NxD arr): N rows of D lists of polynomial degrees
        JacobiParams: arr of parameter tuples [(a,b)] defining Jacobi polynomial 
                      family for each index in PolyIdx[i,:]
    Returns: M 
                  
    """

    N,D = PolyIdxs.shape
 
    # Initialize array for samples 
    samples = np.zeros((M,D)) 

    # Store unique (polynomial part) of indices and their counts
    Uniq_Poly_Indices, Poly_Index_Count = np.unique(PolyIdxs, axis = 0, return_counts = True)

    # Sample M polynomial indices according to their frequency in the approximation space
    sampled_poly_inds = Uniq_Poly_Indices[np.random.choice(np.arange(Uniq_Poly_Indices.shape[0]), M, p = Poly_Index_Count/N)]

    # Store Uniform Random Variables for Importance Sampling
    U = np.random.uniform(0,1,[M,D])

    # Store GQ nodes and induced distributions for Orthogonal Polynomials of Generalized Fourier Coefficients
    Q = 1000
    GQ_Nodes = np.empty((D,Q))
    IDist = np.empty((D, Q, max_poly_degree + 1))
    for i in range(D):
        GQ_Nodes[i], IDist[i] = Jacobi_Induced_Dist(*JacobiParams[i], N=max_poly_degree+1, Q=Q)

    # Construct input samples: 
    for i in range(M):
        for j in range(D):
            deg = int(sampled_poly_inds[i,j])
            samples[i,j] = GQ_Nodes[j][np.digitize(U[i,j], IDist[j][:,deg])]
    
    return samples