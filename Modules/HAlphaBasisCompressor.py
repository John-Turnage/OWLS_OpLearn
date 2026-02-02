import numpy as np

### TO DO: Split into the transform class and weight class. The transform class should take care of the ffts or whatever
### then input into 

class HAlphaBasisCompressor:
    def __init__(self, N, L = 2*np.pi, threshold = 0.99, alpha = 1):
        """
        Parameters:
        - N: Number of grid points per spatial axis (assumes square grid [0,L)^2)
        - L: The length of the square grid
        - threshold: Fraction of average H^alpha energy to retain (e.g., 0.99 for 99%)
        - alpha: The exponent for the H^\alpha space (alpha =1 gives H1, alpha = 0 gives L2,etc )
        """
        self.N = N                 # Number of spatial grid points per axis
        self.L = L                 # Length of single grid axis
        self.threshold = threshold # % of H1 energy to retain in projection
        self.alpha = alpha         # The number of weak derivatives
        self.basis_indices = None  # Indices of modes in reduced basis
        self.M = None              # Number of retained modes
        self.weights = self._compute_weights(N)

    def _compute_weights(self, N):
        """
        Compute the H^alpha weights (1 + k^2 + l^2)^{alpha / 2} over the [0, L)^2 domain.
        """
        L = self.L
        kx = np.fft.fftfreq(N, d=L/N) * L
        ky = kx
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        return np.power(1 + KX**2 + KY**2, self.alpha / 2)

    def fit(self, Y_grid):
        """
        Fit the compressor to training data to determine the reduced basis.

        Parameters:
        - Y_grid: array of shape (num_snapshots, N, N) containing spatial data
        """
        num_snapshots = Y_grid.shape[0]
        N = self.N

        # Accumulate average H^alpha energy per coefficient location
        avg_full_energy = np.zeros(N * N)

        for i in range(num_snapshots):
            dat = Y_grid[i]
            dat_hat = np.fft.fft2(dat) * (self.L / N**2)        # L2-orthonormal FFT
            weighted_flat = (dat_hat * self.weights).flatten()  # HAlpha coefficients
            avg_full_energy += np.abs(weighted_flat)**2

        avg_full_energy /= num_snapshots  # Average over all snapshots

        # Sort coefficient indices by average energy, descending
        sorted_indices = np.argsort(avg_full_energy)[::-1]
        cumulative_energy = np.cumsum(avg_full_energy[sorted_indices])
        total_energy = cumulative_energy[-1]
        energy_fraction = cumulative_energy / total_energy

        # Determine M such that average retained energy >= threshold
        self.M = np.searchsorted(energy_fraction, self.threshold) + 1
        self.basis_indices = sorted_indices[:self.M]

    def project(self, dat):
        """
        Project a single snapshot (N x N) to the reduced H^alpha basis.

        Returns:
        - reduced_coeffs: length-M complex vector of reduced H^alpha coefficients
        """
        if self.basis_indices is None:
            raise ValueError("The compressor must be fit before projecting data.")
        N = self.N
        dat_hat = np.fft.fft2(dat) * (self.L / N**2)
        weighted_flat = (dat_hat * self.weights).flatten()
        return weighted_flat[self.basis_indices]

    def reconstruct_coeffs(self, reduced_coeffs):
        """
        Reconstruct full H^alpha coefficient array from reduced coefficients.

        Returns:
        - full H1 coefficient array of shape (N, N)
        """
        N = self.N
        full_flat = np.zeros(N * N, dtype=complex)
        full_flat[self.basis_indices] = reduced_coeffs
        return full_flat.reshape(N, N)

    def reconstruct_spatial(self, reduced_coeffs):
        """
        Reconstruct the spatial function from reduced H^alpha coefficients.

        Returns:
        - real-valued spatial function (N x N array)
        """
        h_alpha_full = self.reconstruct_coeffs(reduced_coeffs)
        l2_coeffs = h_alpha_full / self.weights
        return np.fft.ifft2(l2_coeffs).real * (self.N**2 / self.L)
    
if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
        
    def test_ON():
        """
        Test to see if unit H^alpha vectors are represented as indicator vectors. 
        """
        
        # Set Parameters 
        N = 200         # Number of grid points in [0,L)
        L = 50          # Length of one axis of grid 
        alpha = -1      # Nubmer of weak derivatives 
        k0, l0 = 1,0    # Wave vector coefficients 

        # Construct Grid
        x = np.linspace(0, L, N, endpoint = False)
        y = x.copy()
        X,Y = np.meshgrid(x,y, indexing = 'ij')

        # Build H^alpha unit vector function 
        Phi = lambda k0,l0 : np.exp(1j *(k0*X + l0*Y) * 2*np.pi/L) / (L* (1 + k0**2 + l0**2)**(alpha/2))
        f = Phi(1,1) + 2*Phi(2,3) + Phi(1,2)
        # Get projection coefficients 
        compressor = HAlphaBasisCompressor(N = N, L=L, threshold = 1.0, alpha = alpha)
        compressor.fit(f[np.newaxis, ...])
        proj = compressor.project(f)

        nonzeros = np.nonzero(proj)[0]
        print("Projecting lin combo of H^alpha Unit Function: (L * (1 + k + l)^(alpha/2))^{-1} * exp(2 pi i / L * (kx + ly))")
        print("f = Phi(1,1) + 2Phi(2,3) + Phi(1,2)")
        print("Nonzero indices in projection:", nonzeros)
        print("Projection vector:", proj)


    def test_reconstruction_accuracy():
        # Set parameters
        N = 128
        L = 2 * np.pi
        alpha = 1.0
        threshold = 0.99

        # Create a  test function
        x = np.linspace(0, L, N, endpoint=False)
        y = x.copy()
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Example function:.
        f = np.sin(X)**2 * np.cos(Y)
        
        # Create training data; here we simply replicate the test function to simulate multiple snapshots.
        training_data = np.array([f for _ in range(2)])
        
        # Instantiate and fit the compressor.
        compressor = HAlphaBasisCompressor(N=N, L=L, threshold=threshold, alpha=alpha)
        compressor.fit(training_data)
        
        # Project the test function onto the reduced basis.
        proj = compressor.project(f)
        
        # Reconstruct the spatial function from the reduced coefficients.
        f_reconstructed = compressor.reconstruct_spatial(proj)
        
        # Compute the relative reconstruction error in the L2 norm.
        error = np.linalg.norm(f - f_reconstructed) / np.linalg.norm(f)
        print("Relative reconstruction error (L2 norm):", error)
        
        # Plot the original and reconstructed functions for visual comparison.
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(f.real, extent=(0, L, 0, L), origin='lower')
        plt.title("Original Function")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(f_reconstructed, extent=(0, L, 0, L), origin='lower')
        plt.title("Reconstructed Function")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    test_ON()
    test_reconstruction_accuracy()


        