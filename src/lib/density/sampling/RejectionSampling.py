import numpy as np

class RejectionSamplerUniform():
    '''Rejection sampler

    This sampler generates sample form a provided proobability distribution 
    using the rejection routine. It generates a unifrm density of points wthin 
    a specified range, and selects points who are greater than a generated random
    number.

    Given a parameter space :math:`\\mathbf \\theta` and a probability density 
    :math:`p( \\mathbf \\theta)`, first samples are generated from a 
    uniform distribution, and for  each :math:`\\theta_i \in \\mathbf \\theta`, 
    a random number :math:`r_i` is generated between 0 and 1 and :math:`\\theta_i` is retained 
    in the list when :math:`r_i < p(\\theta_i)`

    It is important to note that when we do rejectioon sampling, due to the nature of 
    dropping samples, it takes a long time to gather sufficient samples.
    '''
    
    def __init__(self, pdf, ranges=None):
        '''Initialize the Rejection sampler ...
        
        Parameters
        ----------
        pdf : function
            A function that should take an :math:`(N,d)` nd-array and return a uarray :math:`(N,)`.
            The input to the function is thus a set of :math:`N` vectors, each of length :math:`d`.
            This function should return :math:`N` values, each the value of the PDF at the 
            corresponding values of theta.
        ranges : list of lists, optional
            This should represent the lower and upper bounds for each dimension to be sampled, by 
            default `None`. Hence, if there are three dimensions, the list should loook something like
            ``[[0, 1], [5, 6], [0, 1]]``. Here, for example, the second dimension will be 
            sampled between 5 and 6.
        '''
        self.pdf = pdf
        self.ranges = ranges
        if ranges is not None:
            self.d = len(ranges)
        return
    
    def sample(self, N, ranges = None, maxIter=10000):
        '''sample data for the given distribution
        
        Parameters
        ----------
        N : integer
            The number of data points that you wish to sample
        ranges : list of lists, optional
            This should represent the lower and upper bounds for each dimension to be sampled, by 
            default `None`. Hence, if there are three dimensions, the list should loook something like
            ``[[0, 1], [5, 6], [0, 1]]``. Here, for example, the second dimension will be 
            sampled between 5 and 6.
        maxIter : int, optional
            This is the total number of iterations that will be used for sampling. This should be set
            if your region is particularly sparse and the sampling angorithm isnt able to get samples
            with sufficient density values. By default this value is 10000. set this too ``None``
            to turn ooff this feature.

        Returns
        -------
        nd-array :math:`(N,d)`
            The samples from the distribution. Note that, if the ``maxIter`` is too less, then the 
            entire :math:`N` samples might not be returned.
        '''
        
        result = None
        i = 0
        while (result is None) or len(result) < N:
            
            assert not ((ranges is None) and (self.ranges is None)), 'Unspecified range'
            if ranges is None:
                ranges = self.ranges
            self.d = len(ranges)
            lower, upper = zip(*ranges)
            x = np.random.uniform( lower, upper, (N, self.d) )
            p_x = self.pdf(x)

            mask = np.random.uniform(size=N) < p_x
            values = x[mask]
            
            if result is None:
                result = values
            else:
                result = np.vstack((result, values))
                
            i += 1
            if (maxIter is not None) and i >=maxIter:
                break

        result = result[:N]
        return result