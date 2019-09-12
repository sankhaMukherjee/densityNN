import numpy as np

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
        ranges : [type], optional
            [description], by default None
        '''
        self.f = f
        self.p = p
        self.d = None
        self.ranges = ranges
        if ranges is not None:
            self.d = len(ranges)
        return
    
    def integrate(self, N, ranges = None):
        '''[summary]
        
        Parameters
        ----------
        N : [type]
            [description]
        ranges : [type], optional
            [description], by default None
        
        Returns
        -------
        [type]
            [description]
        '''
        
        result = None
        if ranges is None:
            ranges = self.ranges
            
        assert ranges is not None, 'Ranges not provided for uniform sampling'
        d = len(ranges)
        
        rSamples = RejectionSamplerUniform(self.p, ranges)
        samples = rSamples.sample(N)
        result = self.f( samples ).sum()/N
        
        return result
