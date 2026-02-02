import numpy as np
from families import JacobiPolynomials, HermitePolynomials

def Jacobi_Induced_Dist(alpha, beta, N, Q = 1000):
    
    J = JacobiPolynomials(alpha,beta)
    x, w = J.gauss_quadrature(Q)
    V = J.eval(x, range(N))
    W = np.tile(np.sqrt(w), [N, 1]).T * V

    return x, np.cumsum(W**2, axis=0)

def Hermite_Induced_Dist(N = 20, Q = 100):
   
    H = HermitePolynomials()
    x, w = H.gauss_quadrature(Q)
    V = H.eval(x, range(N))
    W = np.tile(np.sqrt(w), [N, 1]).T * V
    
    return x, np.cumsum(W**2, axis=0)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    xJ,FJ = Jacobi_Induced_Dist(np.exp(1), -1/np.pi, N = 200)
    xH,FH = Hermite_Induced_Dist()
    
    plt.figure()
    plt.subplot(121)
    inds = [0,3,7,150]
    plt.plot(xJ, FJ[:,inds])
    plt.legend(["Degree " + str(j) for j in inds], frameon=False)
    plt.axis([-1, 1, 0, 1])
    plt.title("Jacobi: Induced distribution functions for various degrees")

    
    plt.subplot(122)
    inds = [0, 3, 7, 19]
    plt.plot(xH, FH[:,inds])
    plt.legend(["Degree " + str(j) for j in inds], frameon=False)
    plt.axis([np.min(xH), np.max(xH), 0, 1])
    plt.title("Hermite: Induced distribution functions for various degrees")
    plt.show()
