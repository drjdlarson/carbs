from __future__ import annotations
import numpy as np 
from warnings import warn

from serums.models import _DistListWrapper
from serums.models import BaseSingleModel
from serums.models import BaseMixtureModel

""" Importing the classes above will allow us to ensure that these classes can simply be copied and pasted directly into Serums. 
The only additional work that might need to be done here is adding more documentation, or changing variable names. """


class GGIW(BaseSingleModel):
    """Represents a Gamma Gaussian Inverse Wishart Distribution."""
    def __init__(self, alpha:float=None, beta:float=None, mean:np.ndarray=None, covariance:np.ndarray=None, IWdof:float=None, IWshape:np.ndarray = None):
        """Initialize an object. 

        Parameters
        -----------
        alpha : 1 x 1 float
            The Gamma distribution's shape. The default is None.
        beta : 1 x 1 float 
            The Gamma distribution's rate. The default is None.
        mean : N x 1 numpy array
            Mean of the Gaussian distribution. The default is None.
        covariance : N x N numpy array
            Covariance of the Gaussian distribution. The default is None.
        v : 1 x 1 float
            Degrees of freedom for inverse Wishart distribution. The default is None. 
        V : d x d numpy array
            Shape matrix for the inverse Wishart distribution. The default is None. 
        d : 1 x 1 float
            Dimension of extent. It is typically 2 or 3. The default is 2. 
            

        Potential Future Work
        ----------------------
        - Sampling 
        - Integrating scipy?
        - Separate the three different distributions and then have this class handle all of them? 
        
        Right now this class just holds all of the parameters while the mixture / tracker does all the work. 
        The focus of this development is for extended target multi-target tracking, so the sampling / pdf aspects are not developed quite just yet. 

        """
        self.alpha = alpha 
        self.beta = beta
        self.mean = mean
        self.covariance = covariance
        self.IWdof = IWdof
        self.IWshape = IWshape
        if IWshape is not None:
            self.d = IWshape.ndim() 

    @property
    def mean(self):
        """Mean of the gaussian distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.mean

    @mean.setter
    def mean(self, val:np.ndarray):
        self.mean = val

    @property
    def covariance(self):
        """Covariance of the gaussian distribution.

        Returns
        -------
        N x N numpy array.
        """
        return self.covariance

    @covariance.setter
    def covariance(self, val:np.ndarray):
        self.covariance = val

    @property
    def alpha(self):
        """Shape of the gaussian distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self.alpha

    @alpha.setter
    def alpha(self, val:float):
        self.alpha = val

    @property
    def beta(self):
        """Rate of the gamma distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self.beta

    @beta.setter
    def beta(self, val:float):
        self.beta = val

    @property
    def IWdof(self):
        """Degrees of freedom of the inverse Wishart distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self.IWdof

    @IWdof.setter
    def IWdof(self, val:float):
        self.IWdof = val

    @property
    def IWshape(self):
        """Shape of the inverse Wishart distribution.

        Returns
        -------
        d x d numpy array.
        """
        return self.IWshape

    @IWshape.setter
    def IWshape(self, val:np.ndarray):
        self.IWshape = val
        self.d = val.ndim() 

class GGIWMixture(BaseMixtureModel):
    """Gamma Gaussian Inverse Wishart Mixture object."""
    
    def __init__(self, alphas:float=None, betas:float=None, means:np.ndarray=None, covariances:np.ndarray=None, IWdofs:float=None, IWshapes:np.ndarray = None,**kwargs):
        """Initialize a Mixture object. """

        if means is not None and covariances is not None and alphas is not None and betas is not None and IWdofs is not None and IWshapes is not None:
            kwargs["distributions"] = [
                GGIW(alpha=a, beta=b, mean=m, covariance=c, IWdof=v, IWshape=V) for a, b, m, c, v, V in zip(alphas, betas, means, covariances, IWdofs, IWshapes)
            ]
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means for the GGIW components (each is an N x 1 numpy array). Recommended to be read only."""
        return _DistListWrapper(self._distributions, "location")

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn("Must set means to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """List of Gaussian covariances for the GGIW components (each is an N x N numpy array). Recommended to be read only."""
        return _DistListWrapper(self._distributions, "scale")

    @covariances.setter
    def covariances(self, val):
        if not isinstance(val, list):
            warn("Must set covariances to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].covariance = v

    @property
    def alphas(self):
        """List of Gamma alpha parameters for the GGIW mixture components. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "alpha")

    @alphas.setter
    def alphas(self, val):
        if not isinstance(val, list):
            warn("Must set alphas to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].alpha = v

    @property
    def betas(self):
        """List of Gamma beta parameters for the GGIW mixture components. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "beta")

    @betas.setter
    def betas(self, val):
        if not isinstance(val, list):
            warn("Must set betas to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].beta = v

    @property
    def IWdofs(self):
        """List of Inverse Wishart degrees of freedom for the GGIW mixture components. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "IWdof")

    @IWdofs.setter
    def IWdofs(self, val):
        if not isinstance(val, list):
            warn("Must set IWdofs to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].IWdof = v

    @property
    def IWshapes(self):
        """List of Inverse Wishart shape matrices for the GGIW mixture components. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "IWshape")

    @IWshapes.setter
    def IWshapes(self, val):
        if not isinstance(val, list):
            warn("Must set IWshapes to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for _ in range(len(val))]
            self._distributions = [GGIW() for _ in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].IWshape = v
    
    def add_components(self, alphas, betas, means, covariances, IWdofs, IWshapes, weights):
        """Add GGGIW distributions to the mixture."""


        if not isinstance(alphas, list):
            alphas = [
                alphas,
            ]
        if not isinstance(betas, list):
            betas = [
                betas,
            ]
        if not isinstance(means, list):
            means = [
                means,
            ]
        if not isinstance(covariances, list):
            covariances = [
                covariances,
            ]
        if not isinstance(IWdofs, list):
            IWdofs = [
                IWdofs,
            ]
        if not isinstance(IWshapes, list):
            IWshapes = [
                IWshapes,
            ]
        if not isinstance(weights, list):
            weights = [
                weights,
            ]

        self._distributions.extend(
            [GGIW(alpha=a, beta=b, mean=m, covariance=c, IWdof=v, IWshape=V) for a, b, m, c, v, V in zip(alphas, betas, means, covariances, IWdofs, IWshapes)]
        )
        self.weights.extend(weights)

    def get_distribution(self,ii):
        return self._distributions[ii]
