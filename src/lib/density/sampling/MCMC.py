import numpy as np

class MetropolisHastingsNormal:
    '''
        The MH algorithm is sampling implementation of the MCMC algorithm in which we sample 
        form a given distribution :math:`p(\\mathbf x)`. This is done using the following implementation. 
        Given a current position in the parameter space :math:`\\mathbf x`, get a new position using a 
        proposal rule :math:`q(\\mathbf x'| \\mathbf x)`. The probability of acceptance of using this new 
        state is given by the acceptance rate :math:`r`, where:

        .. math::
            :nowrap:

                \\begin{align*}
                r = min(1, \\frac {p(\\mathbf x') q( \\mathbf x | \\mathbf x' )} {p(\\mathbf x) q( \\mathbf x' | \\mathbf x )})
                \\end{align*}


        Since we have both :math:`p(\\mathbf x)` on both the numerator and the demoninator, we can see 
        that we can use an unnormalized density function for :math:`p(\\mathbf x)` and this is still going 
        to work. Typically, for the proposal rule :math:`q`, one chooses a Gaussian distribution. This has 
        two advantages:

        The rule is symmetric. i.e. :math:`q(\\mathbf x | \\mathbf x') = q(\\mathbf x' | \\mathbf x)`
        Its easy to sample from this distribution

        Remember that in this type of sampling, the same point may be used multiple times in sequence. In the 
        limit of infinite samples, this is not a problem. However, this is something that one should remember
        while applying this method. 
    
    '''
    
    def __init__(self, p, d):
        '''Initialize the module
        
        Parameters
        ----------
        p : function
            This function returns the probability density of points within a :math:`d`-dimensional space. Given
            an :math:`(N.d)` dimensional nd-array, consisting of :math:`N` :math:`d`-dimensional vectors, this 
            function is going to return the PDF at each of those N points as a u-array.
        d : int
            This is the dimensionality of the space that we want to sample from.
        '''
        self.p = p
        self.d = d
        self.cov = np.eye(d)
        self.x = np.zeros(d)
        return
    
    def update(self):
        '''update the current point

        The current point in the MCMC chain is stored in the parameter ``x``. Everry time call this functioin, this
        function updated the position of the current point with a potential new point.
        
        Returns
        -------
        uarray
            The current point in :math:`d`-dimensional space
        '''
        
        xD = np.random.multivariate_normal(self.x, self.cov, 1).flatten()
        r = min([self.p(xD)/self.p(self.x), 1])
        if np.random.rand() < r:
            self.x = xD
        
        return self.x
    
    def sample(self, N):
        '''sample form the provided distribution
        
        Parameters
        ----------
        N : int
            the number of samples to generate from the provided distribution
        
        Returns
        -------
        numpy nd-array :math:`(N,d)`
            This is :math:`N` samples form a :math:`d`-dimensional space
        '''
        
        samples = [self.x]
        for _ in range(N-1):
            samples.append( self.update() )
        
        samples = np.array(samples)
        return samples
        
        