# --- Imports --- 
import numpy as np 
import math
import scipy.special as sp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from itertools import product
from typing import Callable, Sequence, Tuple, Any

###-------------------------------------------------------------------------------------------------###
#------------------------ ORTHONORMAL FAMILIES AND QUADRATURE RULES ----------------------------------#
###-------------------------------------------------------------------------------------------------###

# --- Polynomial Evaluations --- 
def eval_orthonormal_jacobi(n: int, x: np.ndarray, jacobi_params: list[float]) -> np.ndarray:
    """
    Evaluate the degree-n orthonormal Jacobi polynomial at x in [-1,1].

    The orthonormal basis is defined with respect to the probability measure:
        dmu(x) = [(1 - x)^a * (1 + x)^b / Z] dx, where Z normalizes the total mass to 1.

    Parameters:
        n (int): Polynomial degree
        x (np.ndarray): Evaluation points
        jacobi_params (list[float]): [a, b] parameters with a, b > -1

    Returns:
        np.ndarray: Values of the orthonormal Jacobi polynomial at x
    """
    alpha, beta = jacobi_params
    c = alpha + beta + 1
    log_num = np.log(2 * n + c) + sp.gammaln(n + c) + sp.gammaln(n + 1)
    log_denom = c * np.log(2) + sp.gammaln(n + alpha + 1) + sp.gammaln(n + beta + 1)
    log_norm = 0.5*(log_num - log_denom)
    norm = np.exp(log_norm)
    Z = 2**(alpha + beta + 1) * sp.beta(alpha + 1, beta + 1)  
    return norm * sp.eval_jacobi(n, alpha, beta, x) * np.sqrt(Z)

def eval_orthonormal_hermite(n: int, x: np.ndarray, hermite_params: list[float]) -> np.ndarray:
    """
    Evaluate the degree-n orthonormal Hermite polynomial under N(mu, sigma**2).

    Parameters:
        n (int): Polynomial degree
        x (np.ndarray): Evaluation points
        hermite_params (list[float]): [mu, sigma**2] where sigma**2 > 0

    Returns:
        np.ndarray: Values of the orthonormal Hermite polynomial at x
    """
    mu, sigma2 = hermite_params
    sigma = np.sqrt(sigma2)
    Hn = sp.eval_hermitenorm(n, (x - mu) / sigma)  
    return Hn / np.sqrt(float(math.factorial(n)))


# --- Underlying Probability Density  ---
def jacobi_weight(x: np.ndarray, jacobi_params: list[float]) -> np.ndarray:
    """
    Evaluate the Jacobi probability density at x in [-1,1].

    Parameters:
        x (np.ndarray): Evaluation points
        jacobi_params (list[float]): [a, b] parameters with a, b > -1

    Returns:
        np.ndarray: Values of the normalized Jacobi density
    """

    alpha, beta = jacobi_params
    Z = 2**(alpha + beta + 1) * sp.beta(alpha + 1, beta + 1)
    return (1 - x)**alpha * (1 + x)**beta  / Z 

def hermite_weight(x: np.ndarray, hermite_params: list[float]) -> np.ndarray:
    """
    Evaluate the N(mu, sigma**2) Gaussian probability density at x.

    Parameters:
        x (np.ndarray): Evaluation points
        hermite_params (list[float]): [mu, sigma**2] where sigma**2 > 0

    Returns:
        np.ndarray: Values of the normalized Gaussian density
    """
    mu, sigma2 = hermite_params
    return np.exp(-(x - mu)**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

# -- Quadrature Rules 
def jacobi_quad(Q: int, jacobi_params: list[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Jacobi quadrature for integrating with respect to the Jacobi probability measure:
        dmu(x) = [(1 - x)^a * (1 + x)^b / Z] dx on [-1, 1]
        where Z is a normalizing constant.

    Parameters:
        Q (int): Number of quadrature nodes
        jacobi_params (list[float]): [a, b] parameters with a, b > -1

    Returns:
        Tuple[np.ndarray, np.ndarray]: (nodes, weights), where weights sum to 1
    """

    alpha, beta = jacobi_params
    nodes, weights = sp.roots_jacobi(Q, alpha, beta)
    Z = 2**(alpha + beta + 1) * sp.beta(alpha + 1, beta + 1)
    return nodes, weights/Z

def hermite_quad(Q: int, hermite_params: list[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Hermite quadrature for integrating against N(mu, sigma**2) probability measure.

    Parameters:
        Q (int): Number of quadrature nodes
        hermite_params (list[float]): [mu, sigma**2] where sigma**2 > 0

    Returns:
        Tuple[np.ndarray, np.ndarray]: (nodes, weights), where weights sum to 1
    """
    mu, sigma2 = hermite_params
    sigma = np.sqrt(sigma2)

    nodes_std, weights_std = sp.roots_hermitenorm(Q)  
    nodes = mu + sigma * nodes_std                   
    weights = weights_std / np.sqrt(2 * np.pi)       

    return nodes, weights

###-------------------------------------------------------------------------------------------------###
###                                     STANDARD SAMPLING 
###-------------------------------------------------------------------------------------------------###

def sample_jacobi(N: int, jacobi_params: list[list[float]]) -> np.ndarray:
    """
    Sample N points from a d-dimensional tensor-product Jacobi distribution on [-1, 1]^d.

    Each dimension uses independent Jacobi parameters [alpha_j, beta_j].

    Parameters:
        N (int): Number of samples
        jacobi_params (list[list[float]]): List of [alpha_j, beta_j] for each dimension

    Returns:
        np.ndarray: Shape (N, d) array of i.i.d. samples
    """
    d = len(jacobi_params)
    samples = np.empty((N, d))

    for j, (alpha, beta) in enumerate(jacobi_params):
        u = np.random.beta(beta + 1, alpha + 1, size=N)
        samples[:, j] = 2 * u - 1  # map [0,1] → [-1,1]

    return samples

def sample_hermite(N: int, hermite_params: list[list[float]]) -> np.ndarray:
    """
    Sample N points from a d-dimensional tensor-product Gaussian distribution N(mu_j, sigma_j**2) in each coordinate.

    Parameters:
        N (int): Number of samples
        hermite_params (list[list[float]]): List of [mu_j, sigma_j**2] for each dimension

    Returns:
        np.ndarray: Shape (N, d) array of i.i.d. samples
    """
    d = len(hermite_params)
    samples = np.empty((N, d))

    for j, (mu, sigma2) in enumerate(hermite_params):
        samples[:, j] = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=N)

    return samples

###-------------------------------------------------------------------------------------------------###
###                               UNIVARIATE INDUCED DISTRIBUTIONS
###-------------------------------------------------------------------------------------------------###

# --- Define Orthonormal Basis --- 
def orthonormal_basis(
    degrees: Sequence[int],
    x: np.ndarray,
    poly_eval: Callable[..., np.ndarray],
    params: list[float]
) -> np.ndarray:
    """
    Evaluate univariate orthonormal polynomials at given points.

    Parameters:
        degrees (Sequence[int]):               Degrees of polynomials (N,)
        x (np.ndarray):                        Evaluation points (Q,)
        poly_eval (Callable[..., np.ndarray]): A function poly_eval(deg, x, params)
        params (list[float]):                  Parameter list, one tuple per dimension

    Returns:
        np.ndarray: Shape (N, Q), each row is p_deg(x)
    """
    
    x = np.asarray(x)
    N = len(degrees)
    P = np.zeros((N, len(x)))
    for i, deg in enumerate(degrees):
        P[i, :] = poly_eval(deg, x, params)
    return P


# --- True Induced Density --- 
def true_induced_density(
    x: np.ndarray,
    degrees: list[int],
    poly_eval: Callable,
    weight_fn: Callable,
    params: list[float]
) -> np.ndarray:
    """
    Evaluate the true univariate induced density at points x for a general orthonormal basis.

    Parameters:
        x (np.ndarray):           Evaluation points
        degrees (list[int]):      Degrees in the index set Lambda
        poly_eval (Callable):     Function (deg, x, params) -> values of orthonormal polynomials
        weight_fn Callable):      Function (x, params) -> weight/density function for orthonormality
        params: (list[float]):    Parameters passed to both poly_eval and weight_fn

    Returns:
        np.ndarray: Induced density evaluated at x
    """
    evals = orthonormal_basis(degrees = degrees, x = x, poly_eval=poly_eval, params=params)
    rho = weight_fn(x, params)
    N = len(degrees)
    return (1.0 / N) * rho * np.sum(evals**2, axis=0)

# --- Univariate Induced Sampling Routine --- 
def induced_sampling(
    num_samples: int,
    degrees: Sequence[int],
    params: list[float], 
    poly_eval: Callable,
    quad_rule: Callable[[int, list[float]], Tuple[np.ndarray, np.ndarray]],
    orthonormal_basis: Callable[[Sequence[int], np.ndarray], np.ndarray] = orthonormal_basis,
    Q: int = 500,
) -> np.ndarray:
    """
    Perform univariate induced sampling from the density:
        dmu_Lambda(x) =  (1/N) sum_{lambda in Lambda} p_lambda(x)^2 dmu(x)

    Parameters:
        num_samples (int):                Number of samples to generate
        degrees (Sequence[int]):          Degrees defining the polynomial subspace Lambda
        params (list[float]):             Parameter list, one tuple per dimension, passed to orthonormal basis and quad_rule
        poly_eval (Callable):             Function (deg, x, params) -> values of orthonormal polynomials
        quad_rule: (Callable)             Function(Q, params) -> [[Q nodes], [Q weights]] 
        Q (int):                          Number of quadrature points used to build the empirical CDF
        
        orthonormal_basis (Callable):     Function orthonormal_basis(degs, x, poly_eval, params) returning vandermond-like matrix

    Returns:
        np.ndarray: Shape (num_samples,), the induced samples 
    """
    nodes, weights = quad_rule(Q, params)

    V = orthonormal_basis(degrees = degrees, x = nodes, poly_eval = poly_eval, params=params)
    cdfs = np.cumsum((V**2) * weights, axis=1)

    sampled_idxs = np.random.randint(0, len(degrees), size=num_samples)
    unifs = np.random.rand(num_samples)

    selected_cdfs = cdfs[sampled_idxs]
    node_idxs = np.array(list(map(partial(np.searchsorted, side='right'), selected_cdfs, unifs)))
    node_idxs = np.minimum(node_idxs, Q - 1)

    selected_nodes = nodes[node_idxs]
    return selected_nodes

###-------------------------------------------------------------------------------------------------###
###                               MULTIVARIATE INDUCED DISTRIBUTIONS
###-------------------------------------------------------------------------------------------------###

#------------------------------------- Define Index Sets ---------------------------------------------#

def tensor_product_index_set(d: int, p: int) -> np.ndarray:
    """Tensor product index set: multi-indices in {0, ..., p}^d"""
    return np.array(list(product(range(p + 1), repeat=d)))

def total_degree_index_set(d: int, p: int) -> np.ndarray:
    """Total degree index set: multi-indices with sum <= p"""
    return np.array([idx for idx in product(range(p + 1), repeat=d) if sum(idx) <= p])

def hyperbolic_cross_index_set(d: int, p: int, q: float = 1.0) -> np.ndarray:
    """
    Construct a hyperbolic cross index set in dimension d,
    with parameter p and parameter q (q=1 by default).
    
    Returns a NumPy array of shape (N, d) where N is the number of indices.
    """
    index_set = []
    for idx in product(range(p + 1), repeat=d):
        prod_term = np.prod([(1 + i)**q for i in idx])
        if prod_term <= (p + 1)**q:
            index_set.append(idx)
    return np.array(index_set)


from collections import deque
def index_set(d, param, max_index = 100000000, weights=None, max_size=100000000, type="hc", p_norm=2):
    """
    Generate an index set in d dimensions with a given criterion:
      - Hyperbolic cross: ∏_{i=1}^d (1+n_i)^(1/weights[i]) <= param+1,
      - lp ball: ((∑_{i=1}^d (1+n_i)^(p_norm/weights[i]))^(1/p_norm)) <= param.

    Parameters:
        d (int): Dimension of the index set.
        param (int): Parameter controlling the size of the index set:
                     for 'hc': corresponds to H_c (with threshold param+1),
                     for 'lp': corresponds to the lp ball radius (with threshold param).
        max_index (int): Maximum value allowed for any index coordinate.
        weights (array-like, optional): Anisotropic weights (each in (0, 1]). If None, defaults to ones.
        max_size (int): Maximum number of indices to generate.
        type (str): Either 'lp' for lp ball or 'hc' for hyperbolic cross.
        p_norm (float, optional): The norm exponent used for the lp ball; only used if type=='lp'. Default is 2.

    Returns:
        np.ndarray: Sorted array of index tuples in the generated index set.
    """
    if weights is None: 
        weights = np.ones(d)
    weights = np.array(weights, dtype=float)
    if weights.shape[0] != d:
        raise ValueError("Length of the weight array must match the dimension d")
    # Avoid zero weights (replace with a very small number)
    weights[weights == 0] = 1e-10
    if not np.all((weights > 0) & (weights <= 1)):
        raise ValueError("All weights must lie in (0,1]")

    index_set = set()
    visited = set()  # Track indices already queued to avoid duplicates
    queue = deque([(0,) * d])
    visited.add((0,) * d)
    
    if type == 'hc':
        threshold = param + 1  # Constant threshold for comparison
    elif type == 'lp':
        threshold = param

    while queue:
        current = queue.popleft()
        
        # Compute the criterion based on the chosen type.
        if type == "hc":
            prod = 1.0
            for i, n in enumerate(current):
                prod *= (1 + n) ** (1.0 / weights[i])
            criterion = prod
        elif type == "lp":
            s = 0.0
            for i, n in enumerate(current):
                s += (n) ** (p_norm / weights[i])
            criterion = s ** (1.0 / p_norm)
        else:
            raise ValueError("Unknown type specified: use 'lp' or 'hc'")

        # Prune if the current index fails the condition
        if criterion > threshold:
            continue

        # Add the current index (if below max_size)
        if len(index_set) < max_size:
            index_set.add(current)
        else:
            break

        # Enqueue neighbors (increment one coordinate at a time, bounded by max_index)
        for i in range(d):
            if current[i] < max_index:
                neighbor = list(current)
                neighbor[i] += 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return np.array(sorted(index_set))

#--------------------------- Define Tensor Product Quadrature Rule ------------------------------------#

def tensor_product_quadrature(
    Q: int,
    d: int,
    params: Sequence[tuple[Any, ...]],
    quad_rule: Callable[[int, tuple[Any, ...]], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Constructs a tensor product quadrature rule in d dimensions via product of Q point univariate quad_rules.

    Returns:
        X: shape (Q^d, d), quadrature nodes
        w: shape (Q^d,), product weights (sum to 1)
    """
    from itertools import product

    # Univariate rules
    univ_nodes = []
    univ_weights = []
    for j in range(d):
        nodes_j, weights_j = quad_rule(Q, params[j])
        univ_nodes.append(nodes_j)
        univ_weights.append(weights_j)

    # Cartesian product of nodes
    grid = list(product(*univ_nodes))
    X = np.array(grid)

    # Tensor product of weights
    weight_grid = list(product(*univ_weights))
    w = np.prod(weight_grid, axis=1)

    return X, w

#------------------------------ Define Multivariate Basis --------------------------------------------#

def eval_multivariate_basis(Lambda: np.ndarray,
                            X: np.ndarray,
                            params: Sequence[tuple[Any, ...]],
                            poly_eval: Callable 
                            ) -> np.ndarray:
    """
    Evaluate multivariate tensor product orthonormal polynomials at m points.

    Parameters:
        Lambda (np.ndarray):    shape (N, d) index set of multi-indices
        X (np.ndarray):         shape (m, d) evaluation points in [-1, 1]^d
        ortho_poly_eval:        call ortho_poly_eval(deg, x, *param)
        params:                 list of params defining orthonormal univariate evaluation 
                                e.g, (alpha, beta) per dimension if jacobi polynomials 

    Returns:
        np.ndarray: shape (N, m) — evaluations of each multivariate basis function at all points
    """
    N, d = Lambda.shape
    m = X.shape[0]
    assert X.shape[1] == d, "X must have same dimension as Lambda"

    evals = np.ones((N, m))

    for j in range(d):
        degs_j = Lambda[:, j]  # shape (N,)
        x_j = X[:, j]          # shape (m,)

        # Evaluate each univariate poly at x_j (returns (N, m))
        univariate_evals = np.array([
            poly_eval(deg, x_j, params[j])
            for deg in degs_j
        ])

        evals *= univariate_evals  # Element-wise product over dimensions

    return evals

#------------------------------ Define True Induced Density -------------------------------------#

def true_induced_density_multivariate(
    X: np.ndarray,
    Lambda: np.ndarray,
    params: Sequence[tuple[Any, ...]],
    poly_eval: Callable,
    weight_fn: Callable 
) -> np.ndarray:
    """
    Evaluate the multivariate induced density at points X using a tensor-product basis.

    Given:
        dmu_Lambda(x) = (1/N) · drho(x) · sum_{lambda in Lambda} |phi_lambda(x)|^2
    where {phi_lambda} is an orthonormal tensor-product basis and
    rho(x) = prod_j rho_j(x_j) is the joint base density.

    Parameters:
        X (np.ndarray):                    Shape (M, d), evaluation points
        Lambda (np.ndarray):               Shape (N, d), multi-index set
        params (list of tuple[Any, ...]):  Parameters for each dimension
        ortho_poly_eval (Callable):        Evaluation routine for univariate orthonormal polynomials
        weight_fn (Callable):              Univariate weight/density function

    Returns:
        np.ndarray: Shape (M,), values of the induced density at each point
    """

    # Evaluate multivariate basis at input points
    V = eval_multivariate_basis(Lambda = Lambda, X = X, params = params, poly_eval = poly_eval)

    # Evaluate product base density rho(x) = prod_j rho_j(x_j)
    M, d = X.shape
    rho = np.ones(M)
    for j in range(d):
        rho *= weight_fn(X[:, j], params[j])

    # Compute induced density
    induced = (1 / Lambda.shape[0]) * rho * np.sum(V**2, axis=0)
    return induced

#------------------------ Define Tensor Product Induced Sampling  ------------------------------------#

def induced_sampling_multivariate(
    num_samples: int,
    Lambda: np.ndarray,
    params: Sequence[tuple[Any, ...]],
    poly_eval: Callable ,
    quad_rule: Callable,
    Q: int = 500
) -> np.ndarray:
    """
    Sample from the true multivariate induced measure by:
        1. Sampling multi-index lambda from Lambda
        2. Drawing x ~ |phi_lambda(x)|^2 dmu(x)
    """
    N, d = Lambda.shape
    assert len(params) == d
    samples = np.zeros((num_samples, d))

    # Sample random multi-indices lambda in Lambda
    sampled_indices = np.random.choice(N, size=num_samples, replace=True)
    sampled_lambdas = Lambda[sampled_indices]

    for j in range(d):
        degrees_j = sampled_lambdas[:, j]
        param_j = params[j]

        univariate_samples = induced_sampling(
            num_samples=num_samples,
            degrees=degrees_j,
            params = param_j,
            poly_eval = poly_eval,
            quad_rule=quad_rule,
            Q=Q,
        )

        samples[:, j] = univariate_samples

    return samples

###-------------------------------------------------------------------------------------------------###
#---------------------------  OPTIMALLY WEIGHTED LEAST SQUARES  --------------------------------------#
###-------------------------------------------------------------------------------------------------###

class Operator_WLS:
    def __init__(self, params, index_set, input_samples, output_samples, poly_eval):
        """
        Initializes the Weighted Least Squares (WLS) operator learning model.
        
        The operator K is approximated as:
            K(f) ≈ tilde K(f) = sum_{n}c_n Phi_n(f)
        where {Phi_n} is the full dictionary of basis functions (indexed by index_set),
        and the coefficients c_n are obtained by solving a weighted least-squares system.
        
        The weights are computed using the full dictionary:
            w_i = N /(sum_n ||Phi_n(f_i)||^2) ,
        ensuring optimal stability.
        
        Parameters:
            params : list
                Parameters for constructing the Jacobi polynomial bases: [(a_i,b_i) for i in range input_dim].
            index_set : ndarray
                Array of shape (N, input_dim) indicating the degree of each basis function.
            input_samples : ndarray
                Training inputs of shape (M, input_dim).
            output_samples : ndarray
                Training outputs of shape (M, out_dim).
            poly_eval: Callable 
        """
        self.params = params
        self.index_set = index_set              # (N, input_dim)
        self.input_samples = input_samples      # (M, input_dim)
        self.output_samples = output_samples    # (M, out_dim)
        
        self.M = input_samples.shape[0]         # Number of training samples.
        self.N = index_set.shape[0]             # Total number of candidate basis functions.
        self.input_dim = input_samples.shape[1]
        self.out_dim = output_samples.shape[1]
        
        # Build the full Vandermonde matrix for training inputs (shape: (M, N)).
        self.V = eval_multivariate_basis(Lambda=self.index_set, 
                                         X = self.input_samples, 
                                         params=self.params, 
                                         poly_eval=poly_eval).T
        
        # Compute the optimal weights for each sample:
        #   w_i = N /(sum_n ||Phi_n(f_i)||^2).
        self.weights = self.N / np.sum(self.V**2, axis=1)
        
        # Form the diagonal scaling matrix D = sqrt(weights/M) (shape: (M, 1)).
        self.D = np.sqrt(self.weights / self.M )[:, np.newaxis]  
        
        # Weighted Vandermonde matrix: A = D * V.
        self.A =  self.D * self.V 
        
        # Solve for the full coefficient matrix via weighted least squares:
        b = self.D * self.output_samples
        self.C = np.linalg.lstsq(self.A,b, rcond = None)[0]  # Matrix of operator coeffs
        
        self.G = self.A.T @ self.A                           # Gram matrix if desired 

       
    def apply(self, new_inputs):
        """
        Apply the learned operator to new inputs.
        
        Parameters:
            new_inputs : ndarray, shape (M_new, input_dim)
        
        Returns:
            approx : ndarray, shape (M_new, out_dim)
                Approximated outputs.
        """
        V_new = eval_multivariate_basis(Lambda=self.index_set, 
                                         X = new_inputs, 
                                         params=self.params, 
                                         poly_eval=poly_eval).T
        approx = V_new @ self.C
        return approx

    def compute_l2_error(self, inputs, true_outputs):
        """
        Compute the L2 error (per sample) between the operator's predictions and true outputs.
        """
        approx_outputs = self.apply(inputs)
        errors = np.linalg.norm(approx_outputs - true_outputs, axis=1)
        return errors
    
    def compute_relative_l2_error(self, inputs, true_outputs):
        """
        Compute the relative L2 error (per sample) between the operator's predictions and true outputs.
        """
        approx_outputs = self.apply(inputs)
        rel_errors = np.linalg.norm(approx_outputs - true_outputs, axis=1) / np.linalg.norm(true_outputs, axis = 1)
        return rel_errors

###-------------------------------------------------------------------------------------------------###
#-------------------------------------  Diagnostics --------------------------------------------------#
###-------------------------------------------------------------------------------------------------###

###---------------------------------
###---------------------------------Univariate Diagnostics
###---------------------------------

###----------------Plot Univariate Basis, Induced Distributions, Induced Density-------------------###
def plot_univariate_diagnostics(
    params,
    degrees,
    poly_eval, 
    quad_rule,
    weight_fn,
    num_samples: int = 300_000,
    Q: int = 500,
) -> None:

    # --- Get Data ---
    nodes, weights = quad_rule(Q, params)
    evals = orthonormal_basis(degrees=degrees, x = nodes, poly_eval = poly_eval, params = params)
    
    induced_samples = induced_sampling(num_samples = num_samples, 
                                       degrees = degrees, 
                                       poly_eval = poly_eval, 
                                       quad_rule = quad_rule, 
                                       params = params,
                                       Q = Q, 
                                       )
    true_induced = true_induced_density(x = nodes, 
                                        degrees = degrees, 
                                        poly_eval = poly_eval, 
                                        weight_fn = weight_fn,
                                        params = params)
                                        
    # --- Plot --- 
    fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (7,7))
    N = len(degrees)
    
    # --- Plot orthonormal (univariate) functions ---
    for d in range(N):
        ax[0,0].plot(nodes, evals[d,:], label = f" Degree {degrees[d]}")
    ax[0,0].legend(loc = 'best', fontsize = 6)
    ax[0,0].set(title = ("Basis"), xlabel = "x", ylabel = r"$p_j(x)$")
    
    # --- Plot Gram matrix of L2 inner products, computed via  Quad ---
    W = np.diag(weights)
    IP = evals @ W @ evals.T
    im = ax[0,1].imshow(IP, origin = 'lower', cmap = 'binary')
    ax[0,1].set(title = r"$\langle p_j, p_k\rangle_{L^2_{\mu}}$", xlabel = r"degree $j$", ylabel = r"degree $k$", xticks = [], yticks = [])
    fig.colorbar(im, ax = ax[0,1], shrink = 0.5)
    
    # --- Plot empirical induced distributions (CDFs), over quadrature nodes ---
    cdfs = np.cumsum((evals**2)*weights, axis = 1)
    for i in range(N):
        ax[1,0].plot(nodes, cdfs[i],"*--", label = f"degree {degrees[i]}", markersize = 0.5, linewidth = 0.2)
    ax[1,0].legend(loc = 'best', fontsize = 6)
    ax[1,0].set(title = "Empirical Induced Distribution", xlabel = "x", ylabel = r"$F_{j}(x)$")
    
    # --- Plot induced density, both empirical and analytic ---
    nbins = int(num_samples/2000)
    ax[1,1].hist(induced_samples, bins = nbins, density = True, color = 'black', alpha = 0.5, label = "Induced Sample Density")
    ax[1,1].plot(nodes, true_induced, color = 'black', label = "True Induced Density")
    ax[1,1].set(title = "Induced Density", xlabel = r"$x$", ylabel = r"$\mathrm{d}\mu_{\Lambda}$")
    ax[1,1].legend(loc = 'best')
                 
    sns.despine(trim = True)
    plt.suptitle(f"Polynomial Class: {str(poly_eval).split()[1].split("_")[-1].capitalize()}")
    plt.tight_layout()
    plt.show()

###---------Check that univariate basis is orthonormal with respect to quad rule-------------------###

def is_orthonormal(
    degrees: Sequence[int],
    poly_eval : Callable,
    quad_rule: Callable,
    params: list[float],
    Q: int = 500,
    tol: float = 1e-8,
    orthonormal_basis: Callable = orthonormal_basis,
) -> bool:
    """
    Check if the basis is orthonormal under the quadrature rule.
    """
    N = len(degrees)
    nodes, weights = quad_rule(Q, params)
    evals = orthonormal_basis(degrees = degrees, x = nodes, poly_eval = poly_eval, params=params)

    W = np.diag(weights)
    IP = evals @ W @ evals.T
    err = np.max(np.abs(IP - np.eye(N)))
    if err >= tol:
        print(f"Basis is not orthonormal (max err = {err:.2e}).")
        return False
    return True

###-----Check that induced (empirical) CDFS are monotonic, and satisfy end point conditions --------###

def is_cdf(
    degrees: Sequence[int],
    quad_rule: Callable,
    poly_eval: Callable,
    params: list[float],
    Q: int = 500,
    left_tol: float = 1e-1,
    right_tol: float = 1e-2,
    orthonormal_basis: Callable = orthonormal_basis
) -> bool:
    """
    Check if the empirical CDFs are valid and monotonic.
    """
    nodes, weights = quad_rule(Q, params)
    V = orthonormal_basis(degrees = degrees, x = nodes, poly_eval = poly_eval, params=params)
    cdfs = np.cumsum(V**2 * weights, axis=1)

    if np.any(cdfs[:, 0] >= left_tol):
        print("Left endpoint of at least one empirical CDF exceeds tolerance.")
        return False
    if np.any(np.abs(cdfs[:, -1] - 1.0) >= right_tol):
        print("Right endpoint of at least one empirical CDF exceeds tolerance.")
        return False
    if not np.all(np.diff(cdfs, axis=1) >= -1e-12):
        print("At least one CDF is not monotonic.")
        return False
    return True

###-----------------------  Univariate Moment Matching Test and Runner -----------------------------###

def univariate_moment_matching_test(
    degrees: Sequence[int],
    test_functions: list[Callable[[np.ndarray], np.ndarray]],
    poly_eval: Callable,
    quad_rule: Callable,
    params: list[float],
    num_samples: int = 200_000,
    Q: int = 500,
    rtol: float = 1e-2
) -> bool:
    """
    Compare empirical and quadrature-based expectations of test functions under the induced measure.

    Returns:
        bool: True if all moment errors are within tolerance.
    """
    # --- Get quadrature nodes and weights
    nodes, weights = quad_rule(Q, params)
    
    # --- Evaluate orthonormal basis on nodes
    V = orthonormal_basis(degrees = degrees, x = nodes, poly_eval=poly_eval, params=params)

    # --- Construct induced density
    # NOTE: the probabiliy weight function is accounted for in the quadrature weights. 
    induced_density = (1 / len(degrees)) *  np.sum(V**2, axis=0) 

    # --- Compute true expectations via quadrature
    true_moments = []
    for f in test_functions:
        f_vals = f(nodes)
        true_expectation = np.sum(f_vals * induced_density * weights)
        true_moments.append(true_expectation)

    # --- Sample from induced measure
    samples = induced_sampling(
        num_samples=num_samples,
        degrees=degrees,
        params=params,
        poly_eval = poly_eval, 
        quad_rule=quad_rule,
        Q=Q,
    )

    # --- Compute empirical expectations
    sample_moments = [np.mean(f(samples)) for f in test_functions]

    # ---  Compare
    print(f"\n{'Moment':<5s} {'True':>8s} {'Sample':>13s} {'Abs. Error':>15s} {'Rel. Error':>14s} ")
    print("-" * 70)
    all_passed = True
    for i, (t, s) in enumerate(zip(true_moments, sample_moments)):
        abs_err = abs(t - s)
        rel_err =  abs_err / abs(t) if abs(t) > 1e-14 else abs(t - s)
        passed = rel_err < rtol
        status = "T" if passed else "F"
        print(f"{f'{i}':<5s} {t:12.4e} {s:12.4e} {abs_err: 12.2e} {rel_err:12.2e} {status}")
        all_passed &= passed

    return all_passed

###----------------------------- Univariate Unit Test Runners --------------------------------------###
    
def univariate_moment_matching_test_runner(
        params: list[float],
        poly_eval,
        quad_rule,
        rel_err_tolerance: float = 1E-3,
        num_samples: int = 1_000_000,
        M: int = 10
    ) -> bool: 
    mom_func = lambda x, m: x**m
    test_functions = [partial(mom_func, m=m) for m in range(M)]
    degrees = np.arange(M)
    print(f"\nRUNNING UNIVARIATE MOMENT MATCHING UNIT TEST:\nParams = {params}\nDegrees = {degrees}\nSamples = {num_samples}")
    return univariate_moment_matching_test(
        degrees = degrees,
        test_functions = test_functions,
        params = params,
        poly_eval = poly_eval,
        quad_rule= quad_rule,
        Q=700,
        num_samples=num_samples,
        rtol=rel_err_tolerance
    )

def run_univariate_tests(
    degrees: Sequence[int],
    params: list[float],
    poly_eval: Callable,
    quad_rule: Callable,
) -> None:
    print(f"\nRUNNING UNIVARIATE UNIT TESTS:\nDegrees = {degrees}\nParams = {params}")

    passed_orth = is_orthonormal(
        degrees=degrees,
        poly_eval=poly_eval,
        quad_rule=quad_rule,
        params=params
    )

    passed_cdf = is_cdf(
        degrees=degrees,
        poly_eval=poly_eval,
        quad_rule=quad_rule,
        params=params
    )

    print(f"{'is_orthonormal':25s} ... {'PASSED' if passed_orth else 'FAILED'}")
    print(f"{'is_cdf':25s} ... {'PASSED' if passed_cdf else 'FAILED'}")

###---------------------------------      
###--------------------------------- Multivariate Diagnostics
###---------------------------------

###---------------------------------- Plot 2D Index Sets -------------------------------------------###

def plot_index_sets(d: int = 2, p: int = 30)->None:
    tensor = tensor_product_index_set(d, p)
    total = total_degree_index_set(d, p)
    hyper = hyperbolic_cross_index_set(d, p, q=1.0)
    
    fig,ax = plt.subplots(ncols = 3, figsize = (8,4))
    strs = ["Tensor Product", "Total Degree", "Hyperbolic Cross"]
    for i, idxset in enumerate([tensor, total, hyper]):
        ax[i].scatter(idxset[:,0], idxset[:,1], color = 'black', s = 2)
        ax[i].set(title = f"{strs[i]}", aspect = 'equal', xticks= [], yticks = [], xlabel = r"$\lambda_1$", ylabel = r"$\lambda_2$")
    plt.suptitle(f"{d}-D Index Sets with Pruning Parameter = {p}", y = 0.85)
    plt.show()


###--------------- Plot Tensor Product Quad Rule for Sanity Check -----------------------------------###

def plot_TP_Quad(params: Sequence[tuple[Any,...]] = [[0,1],[2,3]], 
                 quad_rule = jacobi_quad,
                 Q: int = 200)-> None: 
    d = 2
    X,W = tensor_product_quadrature(Q = Q,d = d, quad_rule = quad_rule, params = params)
    fig,ax = plt.subplots()
    im = ax.scatter(X[:,0], X[:,1], c = W, cmap = 'viridis', s= 0.06)
    ax.set(title = f"Tensor Product Quadrature Rule")
    fig.colorbar(im, ax = ax)
    sns.despine(trim = True)
    plt.show()

###-------------Define test + runner for checking tensor product polynomials are orthonormal--------###

def test_multivariate_orthonormality(
    Lambda: np.ndarray,
    poly_eval: Callable,
    quad_rule: Callable[[int, tuple[Any, ...]], Tuple[np.ndarray, np.ndarray]],
    params: Sequence[tuple[Any, ...]],
    Q: int = 50,
    tol: float = 1e-8,
) -> bool:
    """
    Test if multivariate orthonormal polynomials are orthonormal under tensor product quadrature.

    Parameters:
        Lambda (np.ndarray):         Index set of shape (N, d)
        params (list of tuple):      Polynomial parameters, one tuple per dimension
        Q (int):                     Number of quadrature nodes per dimension
        tol (float):                 Tolerance for orthonormality check
        poly_eval (callable):        Function to evaluate univariate orthonormal polynomials
        quad_rule (callable):        Function returning 1D quadrature nodes and weights

    Returns:
        bool: True if orthonormality condition holds within tolerance
    """
    N, d = Lambda.shape
    X, w = tensor_product_quadrature(Q = Q, d = d, params = params, quad_rule=quad_rule)
    V = eval_multivariate_basis(Lambda = Lambda, X = X, params=params, poly_eval=poly_eval)
    IP = V @ np.diag(w) @ V.T
    err = np.max(np.abs(IP - np.eye(N)))
    if err >= tol:
        print(f"[FAIL] Max deviation from orthonormality: {err:.2e}")
        return False
    return True


# --- Test Orthonormality on given index set

def run_multivariate_orthonormal_tests(
    param_lists: list[list[tuple[Any, ...]]],
    poly_eval: Callable,
    quad_rule: Callable,
    p: int = 3,
    index_fn: Callable[[int, int], np.ndarray] = tensor_product_index_set,
    Q: int = 50,
    tol: float = 1e-8
) -> None:
    """
    Run orthonormality tests for multiple param sets.

    Parameters:
        param_lists (list of list of tuples): Each inner list defines poly parameters per dimension
        p (int):                              Max degree per dimension
        index_fn (callable):                  Function to generate multi-index set
        Q (int):                              Number of quadrature points per dimension
        tol (float):                          Tolerance for orthonormality check
    """
    print("\nRUNNING MULTIVARIATE ORTHONORMALITY TEST:")
    for params in param_lists:
        d = len(params)
        Lambda = index_fn(d, p)
        label = f"d={d}, params={params}"
        print(f"{label:50s} ... ", end="")
        passed = test_multivariate_orthonormality(Lambda=Lambda, params=params, Q=Q, tol=tol, poly_eval=poly_eval, quad_rule=quad_rule)
        print("PASSED" if passed else "FAILED")


###----------------------- Multivariate Moment Matching Unit Test + Runner -------------------------###

def multivariate_moment_matching_test(
    Lambda: np.ndarray,
    test_indices: np.ndarray,
    params: list[tuple[float, float]],
    poly_eval: Callable,
    quad_rule : Callable,
    num_samples: int = 500_000,
    Q: int = 60,
    rtol: float = 1e-2,
    verbose: bool = True
) -> bool:
    """
    Moment test for multivariate induced sampling.

    Parameters:
        Lambda:       Multi-index set used for basis and sampling (N, d)
        test_indices: Multi-index set for test monomials (M, d)
        params:       Jacobi parameters per dimension
        num_samples:  Number of samples for empirical mean
        Q:            Quadrature resolution per dimension
        rtol:         Relative error tolerance

    Returns:
        bool: True if all relative errors are within tolerance.
    """

    d = Lambda.shape[1]
    X_quad, w = tensor_product_quadrature(Q = Q, d = d, params = params, quad_rule = quad_rule)

    # --- True expectations via quadrature
    def monomial_fn(multi_idx):  # multi_idx is shape (d,)
        return lambda x: np.prod(x**multi_idx, axis=1)

    test_funcs = [monomial_fn(alpha) for alpha in test_indices]
    true_moments = [np.sum(f(X_quad) * w) for f in test_funcs]

    # --- Empirical estimate
    samples = induced_sampling_multivariate(
        num_samples=num_samples,
        Lambda=Lambda,
        params=params,
        quad_rule=quad_rule,
        poly_eval=poly_eval,
        Q=Q
    )
    sample_moments = [np.mean(f(samples)) for f in test_funcs]

    # --- Compare
    all_passed = True
    if verbose:
        print(f"\n{'Moment':<8s} {'True':>12s} {'Sample':>12s} {'Abs. Error':>12s} {'Rel. Error':>12s}")
        print("-" * 64)
    for i, (alpha, t, s) in enumerate(zip(test_indices, true_moments, sample_moments)):
        abs_err = abs(t - s)
        rel_err = abs_err / abs(t) if abs(t) > 1e-14 else abs_err
        passed = rel_err < rtol
        if verbose:
            print(f"{str(alpha):<8s} {t:12.4e} {s:12.4e} {abs_err:12.2e} {rel_err:12.2e} {'T' if passed else 'F'}")
        all_passed &= passed

    return all_passed

def multi_mom_match_runner(
    poly_eval: Callable,
    quad_rule: Callable,
    params: Sequence[tuple[any]],
    test_indices: np.ndarray,
    p:int, 
    Q = 400, 
    num_samples = 1_000_0000
)-> bool: 
    print(f"\nRUNNING MULTIVARIATE MOMENT MATCHING UNIT TEST:\nparams = {params}\nQ = {Q}\nnum_samples = {num_samples}\nBasis = TensorProduct({p})")
    Lambda = tensor_product_index_set(d = 2, p = 6)
    return multivariate_moment_matching_test(
            Lambda=Lambda,
            test_indices=test_indices,
            params=params,
            poly_eval=poly_eval,
            quad_rule=quad_rule,
            rtol=1e-2,
            num_samples=num_samples,
            Q = Q
)

###------------Compare 2d induced density: sample hist vs true density -----------------------------###

def compare_multivariate_induced(
    params,
    poly_eval,
    quad_rule,
    weight_fn,
    Lambda = tensor_product_index_set(d=2, p=3),
    log_scale: bool = True,
    contours_hist = True,
    contours_true = True,
    num_samples: int = 300_000,
    grid_size: int = 400,
    eps: float = 1e-12,
    marg_supp = [-1,1]
):

    # --- Sample from induced measure
    samples = induced_sampling_multivariate(
        num_samples=num_samples,
        Lambda=Lambda,
        params = params, 
        poly_eval=poly_eval,
        quad_rule=quad_rule, 
        Q = 500 
    )
    
    # --- Evaluation grid
    x_min, x_max = marg_supp
    x = y = np.linspace(x_min, x_max, grid_size)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    
    # --- True induced density
    Z = true_induced_density_multivariate(X = XY, Lambda = Lambda, params = params, poly_eval = poly_eval, weight_fn=weight_fn).reshape(grid_size, grid_size)
    
    # --- Histogram from samples
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=int(grid_size/4), range=[marg_supp, marg_supp], density=True)

    # --- Colormap normalization
    if log_scale:
        vmin = max(min(Z.min(), H.min()), eps)
        vmax = max(Z.max(), H.max())
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = min(Z.min(), H.min())
        vmax = max(Z.max(), H.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = 'viridis'

    # --- Plotting
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

    # Histogram (empirical)
    axes[0].imshow(H.T, extent=[x_min, x_max, x_min, x_max], origin='lower',
                   cmap=cmap, norm=norm, aspect='equal')
    axes[0].set(title="Induced Density Histogram", xlabel=r"$x_1$", ylabel=r"$x_2$")
    if contours_hist:
        axes[0].contour(
        0.5 * (xedges[:-1] + xedges[1:]),
        0.5 * (yedges[:-1] + yedges[1:]),
        H.T,
        levels=10,
        colors='white',
        linewidths=0.8, 
        alpha = 0.5
    )
    
    # True density
    axes[1].imshow(Z, extent=[x_min, x_max, x_min, x_max], origin='lower',
                   cmap=cmap, norm=norm, aspect='equal')
    axes[1].set(title="True Induced Density", xlabel=r"$x_1$", ylabel=r"$x_2$")
    
    if contours_true:
        axes[1].contour(
        X, Y, Z,
        levels=10,
        colors='white',
        linewidths=0.8,
        alpha = 0.5
    )
    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='bottom', shrink=0.75, pad=0.1, label="Density")
    if log_scale: 
        cbar.set_label("Density (log scale)")

    plt.suptitle(f"Polynomial Class: {str(poly_eval).split()[1].split("_")[-1].capitalize()}   Params: {params}")
    sns.despine(trim=True)
    plt.show()

###---------------------------------      
###--------------------------------- OPTIMALLY WEIGHTED LEAST SQUARES TEST 
###---------------------------------

def test_wls(
                delta: float, 
                eps:float, 
                max_idx_set_param: int,
                params: list[float] = [0,0,1.0], 
                Qs: list[int] =[25,50,100,500,1000], 
                dims:list[float] = [1,5], 
                poly_eval: callable = eval_orthonormal_jacobi, 
                std_sample: callable = sample_jacobi,
                quad_rule: callable = jacobi_quad,
)->None:
    """
    (WLS Theory:) Let eps, delta in (0,1) and  M > c_delta * N * log(2N/eps)
    where c_delta = ((delta + (1-delta)*log(1-delta)))**(-1) 
    then || G - I || < delta with probability 1-eps
    Equivalenlty, cond(A) < sqrt[(1+delta)/(1-delta)] with proba 1-eps, where G= A^TA

    params: 
        delta:      float in (0,1), as in WLS theory above
        eps:        float in (0,1), as in WLS theory above 
        params:     list[int], parameters defining single marginal probabiliy distribution
                        Note: We assume for simplicilty that all such paramters are shared across
                        each marginal in PCE 
        Qs:         list[int], degree of Gauss Quad rule used in Emprical CDF construction needed for 
                        inverse transform sampling of induced distribution. 
                        This serves as a discrization of possible induced samples 
        dims:       list[int], dimension of domain of PCE expansion
        poly_eval   callable, poly_nomial class 
        std_sample  callable, standard sampling for poly_eval class 
        quad_rule   callable, quad rule used to construct emprical CDF 
    """
    print("RUNNING WEIGHTED LEAST SQUARES TEST: ...")
    

    Ns = {}
    conds = {}
    induced_conds = {}
    max_degree = {}

    c_delta =  1.0/(delta + (1-delta)*np.log(1-delta))
    cond_A_bound  = np.sqrt(
        (1+delta)/(1-delta)
    )
    operator = lambda x : x     # irrelevant for computation of the condition number 
    for dim in dims: 

        Ns[dim] = []
        conds[dim] = []
        induced_conds[dim] = {}
        max_degree[dim] = 0
        for Q in Qs:
            induced_conds[dim][Q] = []                   
        
        tp_params = [params for _ in range(dim)]  
        
        for param in range(1,max_idx_set_param):
            idx_set = index_set(d=dim, param = param, weights = np.linspace(1.0,1e-3,dim))
            N = idx_set.shape[0]
            Ns[dim].append(N)
            max_d = np.max(idx_set)

            if max_d > max_degree[dim]:
                max_degree[dim] = max_d

            M = int(c_delta * N * np.log(2*N/eps))

            input_samples = std_sample(M, tp_params)
            output_samples = operator(input_samples)
            conds[dim].append(
                np.linalg.cond(Operator_WLS(params=tp_params, index_set=idx_set,input_samples=input_samples,output_samples=output_samples, poly_eval=poly_eval).A)   
            )

            for Q in Qs:
                input_samples = induced_sampling_multivariate(num_samples=M,Lambda=idx_set,params=tp_params,poly_eval=poly_eval,quad_rule=quad_rule, Q = Q)
                output_samples = operator(input_samples)
                induced_conds[dim][Q].append(
                    np.linalg.cond(Operator_WLS(params=tp_params, index_set=idx_set,input_samples=input_samples,output_samples=output_samples,poly_eval=poly_eval).A)
                )

    fig,ax = plt.subplots(ncols = len(dims), figsize = (12,7), sharey=True)
    for i,dim in enumerate(dims):
        ax[i].plot(Ns[dim], conds[dim], "*--", label = "Standard Sampling", color = 'black')
        for Q in Qs:
            ax[i].plot(Ns[dim], induced_conds[dim][Q],"*--", label = f"Induced Sampling: Q = {Q}")
        ax[i].plot(Ns[dim], np.ones(len(Ns[dim])) * cond_A_bound, color = 'black', label = r"$\sqrt{\frac{1+\delta}{1-\delta}}$")
        ax[i].set(xlabel=rf"$N = \dim \Lambda$  ($\mathrm{{max}}_{{p \in \Lambda_{{N}}}}\deg p$ = {max_degree[dim]})", ylabel = r"$\kappa(\sqrt{{G}})$", title = rf"$\mathrm{{d}}\mu  =  \otimes\mathrm{{d}}\mu^{{({{{params[0]}}}, {{{params[1]}}})}}$ on $\mathbb{{R}}^{{{dim}}}$,", ylim = (0,5))
        ax[i].legend(loc = 'best', fontsize = 8)
        poly_type_str = str(poly_eval).split()[1].split("_")[-1].capitalize()
        title_str = rf"$M = c_{{\delta}} N \log(2N\epsilon^{{-1}}) \rightarrow \mathrm{{Pr}}[\Vert G-I \Vert < \delta]> (1-\epsilon) \leftrightarrow \mathrm{{Pr}}[\kappa(\sqrt{{G}}) < \sqrt{{\frac{{1+\delta}}{{1-\delta}}}}] > (1-\epsilon)$" + "\n" + rf"Polynomial Class = {poly_type_str}, $\delta = {{{delta}}}, \epsilon = {{{eps}}}$"
    plt.suptitle(title_str, y = 1.05)
    sns.despine(trim = True)
    plt.show()

###-------------------------------------------------------------------------------------------------###
###-------------------------------------------------------------------------------------------------###
###-------------------------------------------------------------------------------------------------###

if __name__ == "__main__":
    
    ###-------------------------------------------------------------------------------------------------###
    ###--- Testing Parameters 
    ###-------------------------------------------------------------------------------------------------###

    # --- Define polynomial class used for all tests below
    #poly_type = 'jacobi'
    poly_type = 'hermite'

    # --- For plotting and testing univariate unit tests: 
    # --- i.e, orthonormality, empirical CDFs, visual comparision of induced density with theory, 
    # --- params are also used to ensure that moments of true and emprical inudced densites mathch 
    degrees_uni=[0,1,2,3]        
    params_uni = [0.0, 1.0]

    # --- For chekcing that multivaraite polynomial families are orthonormal. Each dimension should take: 
    # --- [[p1,p2],... [pd,pd]], two parameters for each polynomial class in each dimension.
    # --- for each polynomial class (i.e., row), orthonormality is checked on a tensor product
    # --  index set of max_degree = 3, by default. 
    param_test_multi_ortho = [
        [(0, 0.4), (0, 0.3)],
        [(1, 2), (2, 1)],
        [(0.3, 0.7), (0.5, 0.5)]
    ]

    # --- For comparing the moments under the empircal and true induced density on a given multivaraite polynomial subspace:  
    # --- By default: 
    # ---     The moments are given by a total degree index set with d = 2, p = 4
    # --      The polynomial subspace is defined by a total degree index set of d= 2, p = 6
    params_multi_mom_match  = [(0.2, 1), (0.4, 0.5)]

    # --- For plotting 2D historgram of true and sampled induced density of polynomial class defined by params below
    # --- must choose d = 2 for index set Lambda below, but the pruning param p is flexible. 
    params_compare_multi = [(0, 1), (0, 1)]
    Lambda_compare_multi = tensor_product_index_set(d=2, p=4)

    # --- Parameters needed to define optimally weighted least squres test: see docstring of test_wls
    wls_delta = 0.5
    wls_eps = 0.5
    wls_max_idx_set_param = 50
    wls_Qs = [12,25,50,100,500]
    wls_dims  = [1,3]
    wls_params  = [1,1]

    ###-------------------------------------------------------------------------------------------------###
    ###---  Run Quantitative Unit Tests for Induced Sampling 
    ###-------------------------------------------------------------------------------------------------###
    
    if poly_type == 'hermite': 
        poly_eval = eval_orthonormal_hermite
        quad_rule = hermite_quad
        weight_fn = hermite_weight
        std_sample = sample_hermite
        marg_supp = [-7,7]              # for 2d histogram plotting 
    elif poly_type == 'jacobi':
        poly_eval = eval_orthonormal_jacobi
        quad_rule = jacobi_quad
        weight_fn = jacobi_weight
        std_sample = sample_jacobi
        marg_supp = [-1,1]              # for 2d histogram plotting 
    
    
    #--- Test orthonormal basis and valid CDFs
    run_univariate_tests(
        degrees=degrees_uni, 
        params= params_uni, 
        poly_eval=poly_eval,
        quad_rule=quad_rule
    )

    #--- Test moment matching for univariate case 
    univariate_moment_matching_test_runner(
        M = 10,
        params = params_uni,
        poly_eval=poly_eval, 
        quad_rule=quad_rule,
        rel_err_tolerance =  1E-3
    )

    #--- Test multivariate orthonormal basis 
    run_multivariate_orthonormal_tests(param_lists = param_test_multi_ortho, 
                                       poly_eval = poly_eval, 
                                       quad_rule = quad_rule, 
                                       p=3, 
                                       Q=60)
    
    #--- Test moment matching
    test_indices = total_degree_index_set(d=2, p=4)
    multi_mom_match_runner(poly_eval=poly_eval, 
                           quad_rule=quad_rule,
                           params = params_multi_mom_match,
                           test_indices = test_indices, 
                           p = 6, Q = 200, num_samples = 500_000)

    ###-------------------------------------------------------------------------------------------------###
    ###--- Qualitative Unit Tests for Induced Sampling
    ###-------------------------------------------------------------------------------------------------###
    
    ###--- Visualize various univariate diagonstics 
    plot_univariate_diagnostics(
        params = params_uni,
        degrees =  degrees_uni,
        poly_eval=poly_eval, 
        quad_rule=quad_rule,
        weight_fn=weight_fn,
        num_samples= 300_000,
        Q = 500,
    )
    
    #plot_index_sets()
    #plot_TP_Quad()
    
    ###--- Compare 2d induced sample and true density 
    compare_multivariate_induced(
        Lambda = Lambda_compare_multi,
        params = params_compare_multi,  
        poly_eval=poly_eval, 
        quad_rule=quad_rule, 
        weight_fn=weight_fn,
        log_scale = False,
        contours_hist = False,
        contours_true = True,
        marg_supp = marg_supp)

    ###-------------------------------------------------------------------------------------------------###
    ###--- Run Test for Optimal Sampling Theory for Weighted Least Squares
    ###-------------------------------------------------------------------------------------------------###  

    test_wls(
        delta = wls_delta, 
        eps = wls_eps, 
        max_idx_set_param = wls_max_idx_set_param,
        Qs = wls_Qs, 
        dims = wls_dims,
        params  = wls_params,
        poly_eval = poly_eval, 
        std_sample = std_sample, 
        quad_rule = quad_rule
    )  