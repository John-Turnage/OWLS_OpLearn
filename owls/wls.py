import numpy as np
import sys
sys.path.insert(0, '../Modules/')
from families import JacobiPolynomials
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class Block_WLS:
    def __init__(self, jacobi_params, index_set, input_samples, output_samples, induced):
        """
        Initializes the Weighted Least Squares (WLS) operator learning model.
        
        The operator K is approximated as:
            K(f) ≈ tilde K(f) = sum_{n}c_n Phi_n(f)
        where {Phi_n} is the full dictionary of basis functions (indexed by index_set),
        and the coefficients c_n are obtained by solving a weighted least-squares system.
        
        If induced = True 
            The weights are computed using the full dictionary:
                w_i = N /(sum_n ||Phi_n(f_i)||^2) ,
            ensuring optimal stability.
        Else: 
            The weights are set to one uniformly 
            
        Parameters:
            jacobi_params : list
                Parameters for constructing the Jacobi polynomial bases: [(a_i,b_i) for i in range input_dim].
            index_set : ndarray
                Array of shape (N, input_dim) indicating the degree of each basis function.
            input_samples : ndarray
                Training inputs of shape (M, input_dim).
            output_samples : ndarray
                Training outputs of shape (M, out_dim).
        """
        self.jacobi_params = jacobi_params
        self.index_set = index_set              # (N, input_dim)
        self.input_samples = input_samples      # (M, input_dim)
        self.output_samples = output_samples    # (M, out_dim)
        
        self.M = input_samples.shape[0]         # Number of training samples.
        self.N = index_set.shape[0]             # Total number of candidate basis functions.
        self.input_dim = input_samples.shape[1]
        self.out_dim = output_samples.shape[1]
        self.induced = induced 
        
        # Create JacobiPolynomials for each input dimension.
        self.JacobiPolys = [JacobiPolynomials(*params) for params in self.jacobi_params]
        
        # Build the full Vandermonde matrix for training inputs (shape: (M, N)).
        self.V = self.build_vandermonde(self.input_samples)
        
        # Compute the optimal weights for each sample:
        #   w_i = N /(sum_n ||Phi_n(f_i)||^2).
        if self.induced == True: 
            self.weights = self.N / np.sum(self.V**2, axis=1)
        elif self.induced == False: 
            self.weights = np.ones(self.M)
        
        # Form the diagonal scaling matrix D = sqrt(weights/M) (shape: (M, 1)).
        self.D = np.sqrt(self.weights / self.M)[:, np.newaxis]
        
        # Weighted Vandermonde matrix: A = D * V.
        self.A = self.D * self.V
        
        # Solve for the full coefficient matrix via weighted least squares:
        b = self.D * self.output_samples
        self.C = np.linalg.lstsq(self.A, b, rcond = None)[0]
        
        # Store Gram matrix for condition number calculations, etc 
        #self.G = self.A.T @ self.A
        #self.approx_coeffs = self.C.flatten(order='F')
    
    def build_vandermonde(self, inputs):
        """
        Build the Vandermonde matrix evaluating the polynomial bases at the given inputs.
        
        Parameters:
            inputs : ndarray, shape (M, input_dim)
        
        Returns:
            V : ndarray, shape (M, N)
                The Vandermonde matrix.
        """
        M = inputs.shape[0]
        V = np.ones((M, self.N))
        for n in range(self.N):
            for j in range(self.input_dim):
                deg = int(self.index_set[n, j])
                if deg != 0:
                    poly_eval = self.JacobiPolys[j].eval_1d(inputs[:, j], deg)[:, 0]
                    V[:, n] *= poly_eval
        return V
    
    def apply(self, new_inputs):
        """
        Apply the learned operator to new inputs.
        
        Parameters:
            new_inputs : ndarray, shape (M_new, input_dim)
        
        Returns:
            approx : ndarray, shape (M_new, out_dim)
                Approximated outputs.
        """
        V_new = self.build_vandermonde(new_inputs)
        approx = V_new @ self.C
        return approx
    
    def greedy_reduce(self, test_input_samples, test_output_samples=None, init_size=1, 
                    max_iters=10, top_k=10, plot=True, log=True):
        """
        Perform greedy adaptive basis selection.
        
        Overview:
            1. Precompute full quantities:
                - V_dict: Full Vandermonde matrix (training data).
                - D_full: Diagonal weight matrix from the full operator.
                - V_test_full: Full Vandermonde matrix (test data).
            2. Initialize the active index set with init_size basis functions.
            3. For each iteration:
                a. Extract the reduced Vandermonde matrix V_active corresponding to the active set.
                b. Solve the weighted LS problem on V_active to obtain coefficients C.
                c. Compute training predictions and the residual error.
                d. Using the full weighted matrix from V_dict, compute residual scores for all candidate basis functions.
                e. Select the top_k remaining indices with the highest residual energy.
                f. Evaluate test error using the current active subspace.
                g. Update the active index set (after error evaluation) and repeat.
            4. If the active set equals the full dictionary, the solution matches the full operator.
        
        The weights (D_full) are computed from the full dictionary and are kept constant
        so that the optimal sampling measure is defined on the full dictionary.
        
        Parameters:
            test_input_samples : ndarray, shape (M_test, d_in)
                Test inputs used for evaluation.
            test_output_samples : ndarray, shape (M_test, out_dim), optional
                Test outputs for error computation.
            init_size : int, default 1
                Initial number of active basis functions.
            max_iters : int, default 10
                Maximum iterations for the greedy algorithm.
            top_k : int, default 10
                Number of new indices to add per iteration.
            plot : bool, default True
                If True, plot convergence results.
            log : bool, default True
                If True, output diagnostic print statements.
        
        Returns:
                (active_indices, selected_index_paths, train_errors_list, test_errors_list)
        """
        # Full dictionary: index_set stored from initialization.
        DictIdxs = self.index_set  # (N, d_in)
        N = DictIdxs.shape[0]
        
        # Precomputed full Vandermonde (training) and weight matrix.
        V_dict = self.V            # (M, N)
        D_full = self.D            # (M, 1)
        # Precompute test Vandermonde matrix.
        V_test_full = self.build_vandermonde(test_input_samples)  # (M_test, N)
        
        # ----- Initialize Active and Remaining Indices -----
        ActiveRows = np.arange(init_size)  # Starting active index set.
        RemainingRows = np.setdiff1d(np.arange(N), ActiveRows)
        
        # Lists for tracking over iterations.
        selected_index_paths = []
        train_errors_list = []
        test_errors_list = []
        basis_sizes = []
        residual_scores_matrix = []  # For plotting the evolution of residual scores
        
        if log:
            print("\nBeginning Greedy Adaptive Basis Selection")
        
        # ----- Greedy Loop -----
        for it in range(max_iters):
            if log:
                print(f"--- Iteration {it+1} ---")
            
            # (a) Extract V_active from full Vandermonde.
            ActiveRows = np.unique(ActiveRows)
            V_active = V_dict[:, ActiveRows]     # Shape: (M, N_sub)
            N_sub = V_active.shape[1]
            
            # (b) Solve the weighted LS problem on the active subspace.
            A = D_full * V_active                # (M, N_sub)
            b = D_full * self.output_samples     # (M, out_dim)
            T = A.T @ b                          # (N_sub, out_dim)
            G = A.T @ A                          # (N_sub, N_sub)
            C = np.linalg.solve(G, T)            # (N_sub, out_dim)
            
            # (c) Compute training predictions and residual error.
            pred_train = V_active @ C                    # (M, out_dim)
            residual = pred_train - self.output_samples  # (M, out_dim)
            
            # (d) Compute residual scores using the full weighted dictionary.
            residual_weighted = D_full * residual         # (M, out_dim)
            V_weighted = D_full * V_dict                  # (M, N)

            candidate_projections = V_weighted.T @ residual_weighted  # (N, out_dim)
            # Compute L2 norm for each candidate (across output dimensions).
            scores = np.linalg.norm(candidate_projections, axis=1) / np.linalg.norm(V_weighted, axis = 0) ### COME BACK TO THIS 
            # Record scores over iterations.
            residual_scores_matrix.append(scores)
            
            # (e) Select the top_k new indices from the remaining indices.
            scores_remaining = scores[RemainingRows]
            top_indices = np.argsort(scores_remaining)[-top_k:]
            new_rows = RemainingRows[top_indices]
            
            # (f) Evaluate test error on the active subspace.
            V_test = V_test_full[:, ActiveRows]  # (M_test, N_sub)
            V_test = V_test.reshape(V_test_full.shape[0], -1)  # Ensure 2D shape if N_sub = 1.
            test_pred = V_test @ C                # (M_test, out_dim)
            train_error = np.linalg.norm(pred_train - self.output_samples, axis=1).mean()
            if test_output_samples is not None:
                test_error = np.linalg.norm(test_pred - test_output_samples, axis=1).mean()
            else:
                test_error = np.nan
            
            train_errors_list.append(train_error)
            test_errors_list.append(test_error)
            basis_sizes.append(len(ActiveRows))
            
            if log:
                print(f"Active basis size: {len(ActiveRows)}, Remaining: {len(RemainingRows)}")
                print(f"Mean train error: {train_error:.3e}")
                if test_output_samples is not None:
                    print(f"Mean test error:  {test_error:.3e}")

            # Check if the full dictionary has been recovered.
            if len(ActiveRows) == N:
                if log:
                    print("Full basis recovered. Terminating greedy selection.")
                break
            
            # (g) Update the active index set 
            ActiveRows = np.concatenate([ActiveRows, new_rows])
            RemainingRows = np.setdiff1d(np.arange(N), ActiveRows)
            selected_index_paths.append(ActiveRows.copy())
        
        # ----- Plot convergence results if requested -----
        if plot:
            
            basis_sizes = np.array(basis_sizes)
            train_errors = np.array(train_errors_list)
            test_errors = np.array(test_errors_list)

            # Fit log-log slope for test error
            log_x = np.log(basis_sizes)

            log_y_test = np.log(test_errors)
            slope_test, intercept_test = np.polyfit(log_x, log_y_test, deg=1)
            alpha_test = -slope_test
            ref_line_test = np.exp(intercept_test) * basis_sizes**(-alpha_test)

            # Fit log-log slope for train error
            log_y_train = np.log(train_errors)
            slope_train, intercept_train = np.polyfit(log_x, log_y_train, deg=1)
            alpha_train = -slope_train
            ref_line_train = np.exp(intercept_train) * basis_sizes**(-alpha_train)

            # Plot 
            plt.figure()
            plt.plot(basis_sizes, train_errors, '--o', label="Train Error", color='black')
            plt.plot(basis_sizes, test_errors, '--o', label="Test Error", color='red')
            plt.plot(basis_sizes, ref_line_train, ':', color='gray', label=fr"Train Fit: $\mathcal{{O}}(n^{{-{alpha_train:.2f}}})$")
            plt.plot(basis_sizes, ref_line_test, ':', color='orange', label=fr"Test Fit: $\mathcal{{O}}(n^{{-{alpha_test:.2f}}})$")

            plt.xlabel("Reduced Basis Dimension (Active Index Set Size)")
            plt.ylabel(r"Mean $L^2_{\mathcal{Y}}$ Error")
            plt.yscale('log')
            plt.xscale('log')
            plt.title("Convergence of Adaptive Operator Learning")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

            
            # ----- Plot Residual Heatmap -----
            residual_scores_matrix = np.array(residual_scores_matrix)

            # Flatten to compute percentiles globally
            residuals_flat = residual_scores_matrix.flatten()
            cutoff = 5
            vmin = np.percentile(residuals_flat, cutoff)
            vmax = np.percentile(residuals_flat, 100-cutoff)

            # Avoid vmin being too small for LogNorm
            vmin = max(vmin, 1e-8)

            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
            plt.figure(figsize=(12, 6))
            plt.imshow(residual_scores_matrix, aspect='auto', cmap='inferno', origin='lower', norm = norm)
            plt.colorbar(label=f"Residual Magnitude (Log Scale, ignoring 5% energy tails) ")
            plt.xlabel("Full Dictionary Index")
            plt.ylabel("Iteration")
            plt.title("Residual Energy Distribution Over Iterations")

            # Add horizontal lines between each iteration
            for i in range(1, residual_scores_matrix.shape[0]):
                plt.hlines(i - 0.5, xmin=-0.5, xmax=residual_scores_matrix.shape[1] - 0.5,
                        colors='white', linewidth=0.5, linestyles='-')
            plt.tight_layout()
            plt.show()
        
        # Return the results. If test data is provided, include the best model and best active set.
        
        return ActiveRows, selected_index_paths, train_errors_list, test_errors_list
