from logs import logDecorator as lD
import jsonref
import numpy as np 

config   = jsonref.load(open('../config/config.json'))
logBase  = config['logging']['logBase'] + '.lib.density.linearRegression.BayesianLR'

class BayesianLR_Normal:

    def __init__(self, d, sigmaNoise=1., tau=1):
        '''Initialize a Bayesian linear model with Gausian priors
        
        Parameters
        ----------
        d : int
            the dimensionality of the space
        sigmaNoise : float, optional
            noise sigma, by default 1
        tau : float, optional
            the scale of the initial standard deviation, by default 1
        
        '''

        assert type(d) is int, "The dimension must be an integer"

        self.d = d
        self.sigmaNoise = sigmaNoise
        self.w = np.zeros(self.d)
        self.V = np.eye(self.d)*tau

        return

    def __repr__(self):


        v = '\n'.join([ (f'|     {m}' if i > 0 else f'{m}') for i, m in enumerate(str(self.V).split('\n'))])

        result = ''
        result += f'+----------------------------------------------------------------\n'
        result += f'| Bayesian Linear Regressor, Gaussian Prior, Gaussian Posterior  \n'
        result += f'+----------------------------------------------------------------\n'
        result += f'| w = {str(self.w.flatten())}\n'
        result += f'| V = {v}\n'
        result += f'+----------------------------------------------------------------\n'

        return result

    @lD.log(logBase + '.BayesianLR_Normal.fit')
    def fit(logger, self, XSample, ySample):
        '''[summary]
        
        Parameters
        ----------
        Xsample : [type]
            [description]
        ySample : [type]
            [description]
        '''

        wn, Vn = None, None
        try:
            V0 = self.V.copy()
            w0 = self.w.copy().reshape((-1, 1))
            sigma = self.sigmaNoise

            ySample = ySample.reshape((-1,1))
        
            V0Inv = np.linalg.inv(V0)
            Vn =  sigma**2 * np.linalg.inv( sigma**2 * V0Inv + XSample.T @ XSample)
            wn = Vn @ V0Inv @ w0 + (1/sigma**2) * Vn @ XSample.T @ ySample
            print('Done')
        except Exception as e:
            logger.error(f'Unable to update parameters for the provided data: {e}')
            print(f'Unable to update parameters for the provided data: {e}')
        finally:
            self.V = Vn.copy()
            self.w = wn.copy()

        return

    def predict(self, X):
        '''[summary]
        
        Parameters
        ----------
        X : [type]
            [description]
        '''

        result = X @ self.w.reshape((-1, 1))

        return result