import numpy as np
from families import JacobiPolynomials

class Empirical_Operator_WLS:
    """
    Learns an operator using optimally weighted least squares procedure on an N dimensional 
    subspace of 'rank 1 orthogonal polynomial operators', where the polynomial basis is 
    constructed from empirical data and Jacobi polynomials.

    Attributes:
        input_data (np.ndarray):  Input data of shape (S, d), assumed to be scaled in [-1,1].
        output_data (np.ndarray): Output data of shape (S, m).
        index_set (np.ndarray):   Multi-index array of shape (N, d) defining polynomial basis.
        jacobi_params (list):     Parameters for Jacobi polynomials, one per input dimension.
        induced (bool):           Induced sampling if true: MC sampling if false 
    """

    def __init__(self, input_data, output_data, index_set, jacobi_params, induced = True):
        self.input_data = input_data
        self.output_data = output_data
        self.index_set = index_set
        self.jacobi_params = jacobi_params
        self.S, self.d = input_data.shape
        self.N = index_set.shape[0]
        self.induced = induced

        # Check input range
        eps = 1e-10
        if not (np.all(self.input_data >= -1.0 - eps) and np.all(self.input_data <= 1.0 + eps)):
            raise ValueError("All input data must be scaled to lie in [-1, 1] for Jacobi Polynomial Domain.")

        # Initialize 1D Jacobi polynomial families
        self.jPolys = [JacobiPolynomials(a, a) for a in self.jacobi_params]

        # Placeholders for matrices and coefficients
        self.Q = None                   # Orthonormal Basis for V (training)
        self.R = None                   # Upper triangular matrix from QR
        self.W = None                   # Weight matrix for fit LS Solve
        self.G = None                   # Gram Matrix (W^T @ W) for fit
        self.V = None                   # Vandermonde matrix on training input 
        self.V_pred = None              # Vandermond matrix from last prediction 
        self.Q_pred = None              # Q-basis projection from last prediction 
        self.operator_coeffs = None     # Least Squares Solution (Operator Coeff Expansion)
        self.opt_samples = None         # Indices of Optimal Samples used in WLS 

    def _check_fitted(self):
        """Helper to verify user fits operator before doing anything with it."""
        if self.operator_coeffs is None or self.opt_samples is None:
            raise RuntimeError("Operator must be fitted before calling this method.")

    def _build_weighted_system(self, Vsamp):
        """
        Constructs the weight matrix W and sampling weights for a weighted LS system.

        Args:
            Vsamp (np.ndarray): Sampled rows of Q (orthonormal basis).

        Returns:
            tuple: Weighted matrix W and weights K_opt.
        """
        if self.induced == True: 
            weights = self.N / np.sum(Vsamp**2, axis=1)
        elif self.induced == False: 
            weights = np.ones(Vsamp.shape[0])
        K_opt = weights / Vsamp.shape[0]
        W = (Vsamp.T * np.sqrt(K_opt)).T
        return W, K_opt

    def build_vandermonde(self, X):
        """
        Builds the Vandermonde matrix from Jacobi polynomials on input X.

        Args:
            X (np.ndarray): Evaluation points (n, d).

        Returns:
            np.ndarray: Vandermonde matrix scaled by 1/sqrt(n).
        """
        N = self.index_set.shape[0]
        V = np.ones((X.shape[0], self.N))
        for n in range(self.N):
            for j in range(self.d):
                deg = int(self.index_set[n, j])
                if deg == 0:
                    continue
                V[:, n] *= self.jPolys[j].eval_1d(X[:, j], deg)[:, 0]
        return V / np.sqrt(X.shape[0])

    def fit(self, M):
        """
        Trains the operator using weighted least squares on M optimal samples.

        Args:
            M (int): Number of training samples to use.
        """
        V = self.build_vandermonde(self.input_data)
        Q, R = np.linalg.qr(V)
        self.Q, self.R = Q, R

        K = (self.S / self.N) * np.linalg.norm(Q, axis=1) ** 2
        p = K / np.sum(K)
        if self.induced == True: 
            self.opt_samples = np.random.choice(np.arange(self.S), M, replace=True, p=p)
        elif self.induced == False: 
            self.opt_samples = np.random.choice(np.arange(self.S), M, replace = True)
        Vsamp = np.sqrt(self.S) * Q[self.opt_samples, :]
        W, K_opt = self._build_weighted_system(Vsamp)

        G = W.T @ W   
        g = W.T @ ((self.output_data[self.opt_samples].T * np.sqrt(K_opt)).T)

        self.W = W
        self.G = G 
        self.V = V
        self.operator_coeffs = np.linalg.solve(G, g)
        # For some reason, its faster to use the Gram matrix...
        #self.operator_coeffs = np.linalg.lstsq(self.W, np.sqrt(K_opt)[:,None]* (self.output_data[self.opt_samples]) , rcond = None)[0]

    def predict(self, X):
        """
        Gives output from trained operator from new input data X.

        Args:
            X (np.ndarray): New input data (n, d).

        Returns:
            np.ndarray: Predicted outputs (n, m).
        """
        if self.operator_coeffs is None:
            raise RuntimeError("Operator has not been fitted. Call `fit()` before using `predict()`.")

        V_test = self.build_vandermonde(X)
        Q_test = np.linalg.solve(self.R.T, V_test.T).T
        self.V_pred = V_test
        self.Q_pred = Q_test
        return np.sqrt(X.shape[0]) * Q_test @ self.operator_coeffs

    def predict_on_fit_samples(self):
        """
        Gives output of trained operator on training samples used in the WLS fit.

        Returns:
            np.ndarray: Predicted compressed outputs (M, m).
        """
        self._check_fitted()
        Vsamp = np.sqrt(self.S) * self.Q[self.opt_samples, :]
        return Vsamp @ self.operator_coeffs

    def compute_errors(self, X_eval, Y_eval):
        """
        Computes l2 and relative l2 errors (ground truth vs predicted from 
        output of learned operator) on given evaluation data. 

        Args:
            X_eval (np.ndarray): Evaluation inputs (n, d).
            Y_eval (np.ndarray): True outputs (n, m).

        Returns:
            tuple: (abs_err, rel_err) arrays of shape (n,).
        """
        Y_pred = self.predict(X_eval)
        abs_err = np.linalg.norm(Y_pred - Y_eval, axis=1)
        rel_err = abs_err / np.linalg.norm(Y_eval, axis=1)
        return abs_err, rel_err

    def compute_fit_sample_errors(self):
        """
        Computes errors (ground truth vs output of learned operator) 
        on the training samples used during fitting.

        Returns:
            tuple: (abs_err, rel_err) arrays of shape (M,).
        """
        self._check_fitted()
        Y_true = self.output_data[self.opt_samples]
        Y_pred = self.predict_on_fit_samples()
        abs_err = np.linalg.norm(Y_pred - Y_true, axis=1)
        rel_err = abs_err / np.linalg.norm(Y_true, axis=1)
        return abs_err, rel_err
