import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def sine_basis_vandermonde(x, N):
    """
    Evaluates 

        sin(j pi x),   j \in [N],

    as a x.size x N array.
    """
    assert N > 0, "N must be a positive integer"
    return np.sin(np.pi*np.outer(x.flatten(), np.arange(1, N+1)))

def cosine_basis_vandermonde(x, N):
    """
    Evaluates 

        cos(j pi x),   j \in [N]_0,

    as a x.size x (N+1) array.
    """
    assert N > 0, "N must be a positive integer"
    return np.cos(np.pi*np.outer(x.flatten(), np.arange(N+1)))

def fourier_quad_rule(M,interval=[0,1], prob_measure=False):
    """
    Generates an M-point Fourier quadrature rule (equispaced, equally weighted)
    on the specified interval. If prob_measure is set to True, then the weights
    sum to 1.
    """

    assert M > 0, "M must be a positive integer"
    x = np.linspace(interval[0], interval[1], M+1)[:M]
    w = np.ones(M)/M
    if not prob_measure:
        w *= (interval[1] - interval[0])

    return x, w

def Viscous_Burgers_Spectral_Galerkin(v, dt, M, U0):
    """
    Computes the solution to,

                            u_t + uu_x = v u_{xx},       x in [0,1]
                               u(-1,t) = u(1,t) = 0,    t > 0
                                u(x,0) = u0

    through a spectral Galerkin procedure with ansatz of the form, 

                            u(x,t) = sum_{j=0,N-1} U_j(t) sqrt(2) *sin(j pi x)
                                  
    The linear term is treated implicitly with Backward Euler and the nonlinear
    term is treated explicitly with Forward Euler. The PDE is integrated up to
    terminal time T = M*dt.

    Requires input U_j(0) is given (as U0), from which N is inferred. U0 can be
    a 2-d array, in which case U0[k] for each k is a single initial state. 

    The output U is a K x N array, where U[k,:] is the (Galerkin)
    solution vector for the k'th input initial data at time T.
    t[m].

    If U0 is a vector, then the corresponding dimensions of U are squeezed out.
    """

    if len(U0.shape) == 1:
        N = U0.size
        K = 1
    elif len(U0.shape) == 2:
        K, N = U0.shape

    U = np.zeros([K, N])
    U[:,:] = U0

    # The update is 
    # u^j_{n+1} * (1 + dt*v*(j*pi)**2) = u^j_n - dt*(u*u_x)_j

    # Implicit spectral (second) differentiation for linear term:
    d2invmat = 1/(1 + dt*v*(np.pi*np.arange(1, N+1))**2)

    ## For explicit spectral differentiation of nonlinear term:

    # Nodal differentiation evaluation
    x, w = fourier_quad_rule(4*N+1, interval=[0,1])
    d1_nodal_eval = np.sqrt(2)*cosine_basis_vandermonde(x,N)[:,1:]
    d1_nodal_eval *= np.pi*np.arange(1, N+1)

    # Nodal evaluation
    V = np.sqrt(2)*sine_basis_vandermonde(x,N)
    wV = (V.T * w).T

    # Evaluting u u_x in spectral space is:
    # ( (U @ V.T) * (U @ d1_nodal_eval.T) ) @ wV

    for m in range(M):
        U = (U - dt * ( (U @ V.T) * (U @ d1_nodal_eval.T) ) @ wV) * d2invmat

    return U


if __name__ == "__main__":

    N = 100
    U0 = np.zeros([4, N])
    U0[0,0] = -1.
    U0[1,1] =  1.
    U0[2,2] = -1.
    U0[3,:] = np.random.randn(N)/np.abs(np.arange(1, N+1))

    t = 0.3
    v = 0.01
    dt = 0.001

    U1 = Viscous_Burgers_Spectral_Galerkin(v, dt, int(t/dt), U0[:,:10])
    U2 = Viscous_Burgers_Spectral_Galerkin(v, dt, int(t/dt), U0)

    M = 100
    x = np.linspace(0,1,M)
    V = np.sqrt(2)*sine_basis_vandermonde(x, N)

    P = U0.shape[0]
    plt.figure()
    for plotid in range(P):
        plt.subplot(1, P, plotid+1)
        plt.plot(x, V @ U0[plotid], 'b',
                 x, V @ U2[plotid], 'r')
        plt.xlabel('$x$')
        plt.ylabel('$u(x,t)$')
        if plotid==0:
            plt.legend(['Time 0', 'Time {0:f}'.format(t)], frameon=False)

    plt.show()

