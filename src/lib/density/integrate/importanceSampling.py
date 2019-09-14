import numpy as np
from lib.density.sampling import RejectionSampling as RS
from lib.density.utils import OptimizeNormalKL
from scipy.stats import multivariate_normal


class ImportanceSampleIntegrateUniform():
    '''Importance Sampling is the idea that we want to solve the integral of the form:

        .. math:: \\mathbb E(f) = \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  d \\mathbf w

        It is entirely possible that one samples from the distribution :math:`p(\mathbf w)`. 
        Under this formulation, the equation for importance sampling (in the limit of 
        infinite samples :math:`N`) would be given by the following:

        .. math:: \\mathbb E(f) = \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  d \\mathbf w 
        .. math:: \\mathbb E(f) = \\frac 1 N \\sum_{i=1}^N {f(w_i)}, w_i \\sim p(\\mathbf w)
        
    '''
    
    def __init__(self, f, p, ranges=None):
        '''initialize the Importance Sampler.
        
        Parameters
        ----------
        f : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that needs to be integrated over the :math:`d`
            dimensional vector space.
        p : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that represents the probability density function
            over the :math:`d` dimensional space.
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        '''
        self.f = f
        self.p = p
        self.d = None
        self.ranges = ranges
        if ranges is not None:
            self.d = len(ranges)
        return
    
    def integrate(self, N, ranges = None):
        '''Integrate the function ``f`` over the :math:`d` dimensional space. 

        The integration is performed as a sum with :math:`N` points sampled throough
        rejection sampling with the probability density ``p``.
        
        Parameters
        ----------
        N : int
            Number of points to use as samples for the integration
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        
        Returns
        -------
        float
            The result of the integration.
        '''
        
        result = None
        if ranges is None:
            ranges = self.ranges
            
        assert ranges is not None, 'Ranges not provided for uniform sampling'
        d = len(ranges)
        
        rSamples = RS.RejectionSamplerUniform(self.p, ranges)
        samples = rSamples.sample(N)
        result = self.f( samples ).sum()/N
        
        return result

class ImportanceSampleIntegrateNormal():
    '''Importance Sampling is the idea that we want to solve the integral of the form:

        .. math:: \\mathbb E(f) = \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  d \\mathbf w

        We shall sample directly using the multivariate Gaussian distribution that we know, rather than using 
        uniform sampling. That should significantly reduce the amount of time required. However for this we shall 
        need to change the equations a bit.

        .. math::
            :nowrap:

                \\begin{align*}
                \\mathbb E(f) &= \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  \\frac {q(\\mathbf w)} {q(\\mathbf w)} d \\mathbf w \\\\
                              &= \\int_{-\\infty}^{\\infty} \\Big( f(\\mathbf w)  \\frac {p(\\mathbf w)} {q(\\mathbf w)} \Big) q(\\mathbf w)  d \\mathbf w \\\\
                              &= \\frac 1 N \\sum_{i=1}^N w_s{f(w_i)}, w_i \\sim q(\\mathbf w)
                 \\end{align*}


        The important thing about this transformation is that the sampling form another distribution (like :math:`q(\mathbf w)`) is typically easier to do.

        For unnormalized distributions, we can use the following formula instead:

        .. math::
            :nowrap:

                \\begin{align*}
                \\mathbb E(f) &= \\frac 1 {\\sum_{i=1}^N w_s} \\sum_{i=1}^N w_s{f(w_i)}, w_i \\sim q(\\mathbf w)
                \\end{align*}

    '''
    
    def __init__(self, f, p, ranges=None):
        '''initialize the Importance Sampler.
        
        Parameters
        ----------
        f : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that needs to be integrated over the :math:`d`
            dimensional vector space.
        p : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that represents the probability density function
            over the :math:`d` dimensional space.
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        '''
        self.f = f
        self.p = p
        self.d = None
        self.ranges = ranges
        if ranges is not None:
            self.d = len(ranges)
        return
    
    def integrate(self, N, ranges = None):
        '''Integrate the function ``f`` over the :math:`d` dimensional space. 

        The integration is performed as a sum with :math:`N` points sampled throough
        rejection sampling with the probability density ``p``.
        
        Parameters
        ----------
        N : int
            Number of points to use as samples for the integration
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        
        Returns
        -------
        float
            The result of the integration.
        '''
        
        result = None
        if ranges is None:
            ranges = self.ranges
            
        assert ranges is not None, 'Ranges not provided for uniform sampling'
        d = len(ranges)
        mean = np.zeros(d)
        cov  = np.eye(d)*1000
        
        samples = np.random.multivariate_normal(mean, cov, N)
        p = self.p(samples)
        q = multivariate_normal(mean, cov).pdf(samples)
        f = self.f(samples)
        
        result = np.exp(np.log(f) + np.log(p) - np.log(q))
        result = result.sum()/N
        
        return result

class ImportanceSampleIntegrateNormalAdaptive():
    '''Importance Sampling is the idea that we want to solve the integral of the form:

        .. math:: \\mathbb E(f) = \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  d \\mathbf w

        Rather than uniform sampling, it is possible that we use a Gaussian distribution to sample form. This
        can be easily accomplished if we use a Gaussian distribution :math:`q` that closely represents our 
        required distribution :math:`p`. This can be donee by finding the parameters of a multivariate Gaussian
        distribution that closely matches the required distribution using the KL-divergence.
        
        We shall sample directly using the multivariate Gaussian distribution that we know, rather than using 
        uniform sampling. That should significantly reduce the amount of time required. However for this we shall 
        need to change the equations a bit.

        .. math::
            :nowrap:

                \\begin{align*}
                \\mathbb E(f) &= \\int_{-\\infty}^{\\infty} f(\\mathbf w) p(\\mathbf w)  \\frac {q(\\mathbf w)} {q(\\mathbf w)} d \\mathbf w \\\\
                              &= \\int_{-\\infty}^{\\infty} \\Big( f(\\mathbf w)  \\frac {p(\\mathbf w)} {q(\\mathbf w)} \Big) q(\\mathbf w)  d \\mathbf w \\\\
                              &= \\frac 1 N \\sum_{i=1}^N w_s{f(w_i)}, w_i \\sim q(\\mathbf w)
                 \\end{align*}


        The important thing about this transformation is that the sampling form another distribution (like :math:`q(\mathbf w)`) is typically easier to do.

        For unnormalized distributions, we can use the following formula instead:

        .. math::
            :nowrap:

                \\begin{align*}
                \\mathbb E(f) &= \\frac 1 {\\sum_{i=1}^N w_s} \\sum_{i=1}^N w_s{f(w_i)}, w_i \\sim q(\\mathbf w)
                \\end{align*}

    '''
    
    def __init__(self, f, p, ranges=None):
        '''initialize the Importance Sampler.
        
        Parameters
        ----------
        f : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that needs to be integrated over the :math:`d`
            dimensional vector space.
        p : function
            This takes a numpy nd-array of dimensions :math:`(N,d)` representing a set of :math:`N` vectors 
            of dimension :math:`d` and return :math:`N` values that represents the probability density function
            over the :math:`d` dimensional space.
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        '''
        self.f = f
        self.p = p
        self.d = None
        self.ranges = ranges
        self.optKL = None
        if ranges is not None:
            self.d = len(ranges)
        return
    
    def integrate(self, N, ranges = None):
        '''Integrate the function ``f`` over the :math:`d` dimensional space. 

        The integration is performed as a sum with :math:`N` points sampled throough
        rejection sampling with the probability density ``p``.
        
        Parameters
        ----------
        N : int
            Number of points to use as samples for the integration
        ranges : list of lists, optional
            This is a list of :math:`d` ranges, one for each dimension over which the :math:`d` dimensional vector
            space is going to be uniformly sampled, by default ``None``
        
        Returns
        -------
        float
            The result of the integration.
        '''
        
        result = None
        if ranges is None:
            ranges = self.ranges
            
        assert ranges is not None, 'Ranges not provided for uniform sampling'
        self.d = len(ranges)
        
        
        if self.optKL is None:
            lower, upper = zip(*ranges)
            mu_0 = np.zeros(self.d)

            tempSamples = np.random.uniform(lower, upper, (1000, self.d))
            pProb       = self.p(tempSamples)
            self.optKL = OptimizeNormalKL.OptimizeNormalKL(pProb, tempSamples)
            self.optKL.optimize(mu_0)

        samples = np.random.multivariate_normal( self.optKL.mu, self.optKL.sigma, N)
        p = self.p(samples)
        q = self.optKL.pdf( samples )
        f = self.f(samples)
        
        #result = np.exp(np.log(f) + np.log(p) - np.log(q))
        result = f * (p/q)
        result = result.sum()/N
        
        return result