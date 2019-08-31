from logs import logDecorator as lD
import jsonref
import numpy as np

config   = jsonref.load(open('../config/config.json'))
logBase  = config['logging']['logBase'] + '.lib.density.utils.OptimizeNormalKL'

class OptimizeNormalKL:
    '''Generate a Normal distribution using the KL divergence
    
    This class allows you to generate a normal distribution that
    can be later used as a proxy for a different distribution. 
    This can be used in many applications, such as MCMC integration
    of distributions when the actual distribution is not known. In
    this system, you will start by defining :math:`\mathbf \\theta` and 
    :math:`p`. 
    
    Here, :math:`\mathbf \\theta = [\\theta_1, \\theta_2, \\ldots, \\theta_N]` 
    is represented by an :math:`N \\times d` array, where each row is a single
    :math:`d`-dimensional :math:`\\theta` vector.

    :math:`p` is an :math:`N`-dimensional vector that represents the precomputed
    values of the probability densities at each of the :math:`N` points of the 
    :math:`\\theta` vactors.

    This class will attempt to generate a new multinomial Gaussian Distribution
    :math:`q`, where
    
    .. math:: q(\\theta) = \\frac {1} {\\sqrt{ (2 \\pi) ^k  |\\Sigma|}} exp \\Big( -\\frac 1 2  (\\theta-\\mu)^{\\top} \\Sigma^{-1} (\\theta-\\mu)  \\Big)

    such that the KL-divergence

    .. math:: D_{KL}(p||q) = - \sum_{\\theta \in \chi} p(\\theta) \log \\frac {q(\\theta)} {p(\\theta)}

    is minimized. So, after the optimization process, we should have a :math:`\\mu` and a :math:`\\Sigma` that
    will minimize the KL-divergence. For simplicity, this will only generate a diagonal :math:`\\Sigma`.

    '''
    
    def __init__(self, p, theta):
        '''Create an instance of the object
        
        Parameters
        ----------
        p : (N,) uarray
            the probability assocoated with each value of ``theta`` that we are trying
            to replicate
        theta : (N,d) nd-array
            A set of ``N`` values of ``d``-dimensional ``theta`` values over which we shall
            find a multivariate normal distribution over.
        '''
        
        self.p = p
        self.theta = theta
        self.d = theta.shape[1]
        self.mu = None
        self.sigma = None
        self.q = None
        
        return

    
    def minFunc(self, x):
        '''internal function: not to be used
        
        Parameters
        ----------
        x : (2d,) nd-array
            the mean and the std of the arrays that we are trying
            to optimize coalesed into a signle array
        
        Returns
        -------
        float
            The value of the KL divergence for the supplied value of
            the mu and theta between the provided probability distribution
            and our approximate probability distribution
        '''
        
        mu = x[:self.d].reshape(1, -1)
        sigma = x[self.d:]#.reshape(1, -1)
        sigma_1 = np.eye(self.d)
        for i, s in enumerate(sigma):
            sigma_1[i,i] = s
        
        q = multivariate_normal.pdf( theta, mu, sigma_1 )
        result = divergences.D_KL( self.p, q)
        
        return result
    
    def optimize(self, mu0, *args, **kwargs):
        '''function that is used for the optimization process

        This is going to get optimized values of the :math:`\\mu` and the 
        :math:`\\Sigma` that would allow a multivariate normal distribution
        to be generated that will minimize the KL divergence between the
        provided data and the generated multivariate normal distribution.
        
        Parameters
        ----------
        mu0 : (d,) nd-array
            The initial guess of the value of the :math:`\\mu`
        
        Returns
        -------
        (N,) uarray
            The probability of the values for the normal distribution at the
            specified points
        '''
        
        sigma0 = np.ones( mu0.shape )
        x0 = np.hstack( (mu0, sigma0) )
        
        result = minimize( self.minFunc, x0, *args, **kwargs )
        x = result['x']
        
        self.mu = x[:self.d].reshape(1, -1)
        sigma = x[self.d:]
        self.sigma = np.eye(self.d)
        for i, s in enumerate(sigma):
            self.sigma[i,i] = s
        
        self.q = multivariate_normal.pdf( self.theta, self.mu, self.sigma )
        
        
        return self.q
