from logs import logDecorator as lD
import jsonref
import numpy as np

config   = jsonref.load(open('../config/config.json'))
logBase  = config['logging']['logBase'] + '.lib.density.utils'

@lD.log( logBase + '.D_KL' )
def D_KL(logger, p, q, epsilon=1e-50, returnVals=False):
    '''the KL divergence of ``p`` and ``q``

    This function returns the KL divergence given by 

    .. math:: D_{KL}(p||q) = - \sum_{x \in \chi} p(x) \log \\frac {q(x)} {p(x)}
    
    Parameters
    ----------
    logger : logging.logger instance
        Logging instance that is used for the logging errors in case there are any.
    p : nd-array
        Values of the probabbility density ``p`` evaluated at different instances of 
        ``x``. Note that for calculating the KL divergence, the actual values of
        ``x`` are not needed. Only the densities at ``x``. Ideally, this needs to
        sum to zero, since this is a probability density. However, no check is 
        performed to make sure that this is true.
    q : nd-array (the same shape as p)
        Values of the probabbility density ``q`` evaluated at different instances of 
        ``x``. Note that for calculating the KL divergence, the actual values of
        ``x`` are not needed. Only the densities at ``x``. There should be a one-
        to-one correspondence between the points of ``x`` between ``p`` and ``q``.
    epsilon : float, optional
        A small number that will disallow the probability to reduce to zero, by default 1e-50
    returnVals : bool, optional
        Optionally also return the KL divergence values at each position of the array, which
        can sometimes be useful for plotting or error correction, by default False
    
    Returns
    -------
    float
        The KL divergence. In case an error occurs, ``None`` is returned, and the error logged.
    '''

    try:
        p_1 = np.clip(p, epsilon, None)
        q_1 = np.clip(q, epsilon, None)
        
        divergence = p_1 * ( np.log(p_1) - np.log(q_1) )
        
        if returnVals:
            return np.sum(divergence), divergence
        
        return np.sum(divergence)
    except Exception as e:
        logger.error(f'Unable to generate the KL divergence: {e}')


    return None