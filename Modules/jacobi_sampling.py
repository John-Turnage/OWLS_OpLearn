import numpy as np

def jacobi_samp(a,b,M):
    return 1 - 2*np.random.beta(a+1,b+1,M)

def jacobi_tensor_samp(jacobi_params, M):
    """
    jacobi_params = [[a,b] for params in tensor dimension] (list of float tuples)
    M  = number of samples (int)
    """
    dim = len(jacobi_params)
    samples = np.empty((M,dim))
    for d in range(dim):
        samples[:,d] = jacobi_samp(*jacobi_params[d], M)
    return samples

if __name__ == "__main__": 
    from scipy.special import beta as beta_func
    from scipy.integrate import quad
    import matplotlib.pyplot as plt

    def dJ(a,b,x):
        """returns the normalized Jacobi weight function"""
        return ((1-x)**a)*((1+x)**b)/((2**(a+b+1) * beta_func(a + 1, b + 1)))

    def dB(a,b,x):
        "returns the Beta distribution"
        return ((x**(a-1)) * ((1-x)**(b-1))) / beta_func(a, b)

    # By a change of variables, its easy to see that dJ(a,b,x) = 0.5*dB(a+1,b+1,(1-x)/2)
    # Here we plot this for a few instances of alpha = beta 
    fig,axs = plt.subplots(1,2)

    x = np.linspace(-1,1,100)
    for a in range(0,550,50):
        I = quad(lambda x : dJ(a,a,x), -1,1)
        axs[0].plot(x,dJ(a,a,x),label = "a = {a}".format(a=a))
        axs[0].plot(x,0.5*dB(a+1,a+1,(1-x)/2), 'k--', alpha = 0.3)
    axs[0].legend(loc = 'best', fontsize = 8)
    axs[0].set_title(r"Check: $dJ^{\alpha,\alpha}(x) = \frac{1}{2}dB^{\alpha + 1, \alpha +1}(\frac{1-x}{2})$ ", fontsize = 8)

    a = b = 100
    N = 100000
    nBins = 500

    bsamps = np.random.beta(a+1,b+1,N)
    xsamps = (1-2*bsamps)

    counts,bins = np.histogram(xsamps,nBins, density = True)
    plt.hist(bins[:-1], bins,weights = counts)
    axs[1].plot(x, dJ(a,b,x), label = r"$dJ^{\alpha,\alpha}(x)$")
    axs[1].legend(loc = 'best', fontsize = 8)
    axs[1].set_title("{N} ".format(N=N) + r"normalized samples from $1-2u$, with" +"\n" +r"$u\sim \beta^{\alpha + 1, \alpha +1}$ for $\alpha$ = " + "{a}".format(N = N, a= a ), fontsize = 8)
    plt.show()


    # Now test main sampling function:
    M = 100000
    dim = 4 
    jacobi_params = [[20*(a**(1.5)),20*(a**(1.5))] for a in range(dim)]

    samps = jacobi_tensor_samp(jacobi_params, M)
    nBins = 500
    x = np.linspace(-1,1,1000)
    indxs = [[n,m] for n in range(2) for m in range(2)]

    fig,axs = plt.subplots(2,2)
    fig.suptitle("{M}".format(M=M) + r" Normalized Samples and Exact Marginal Densities from $\otimes_{i=1}^4 dJ^{\alpha_i,\alpha_i}$ for $\alpha_i = 20i^{1.5}$", fontsize = 8)
    for d in range(dim):
        ax = axs.flat[d]
        s = samps[:,d]
        a, b = jacobi_params[d]
        counts,bins = np.histogram(s,nBins, density = True)
        ax.hist(bins[:-1], bins, weights = counts)
        ax.plot(x, dJ(a,b,x), label = r"$dJ^{\alpha,\alpha}(x)$")
        ax.legend(loc = 'best', fontsize = 8)
        ax.set_title(r"$\alpha$ = "+"{a}:".format(a=a), fontsize = 8)
    
    axs[0,1].set_xticklabels([])
    axs[0,0].set_xticklabels([])

    plt.show()
        