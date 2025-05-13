from __future__ import annotations
import numpy as np 
from warnings import warn

import matplotlib.pyplot as plt

import scipy.stats as stats
from matplotlib.patches import Ellipse

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
        self._alpha = alpha 
        self._beta = beta
        self._mean = mean
        self._cov = covariance
        self._IWdof = IWdof
        self._IWshape = IWshape 
        if IWshape is not None:
            self._d = IWshape.shape[0]

    @property
    def mean(self):
        """Mean of the gaussian distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self._mean

    @mean.setter
    def mean(self, val:np.ndarray):
        self._mean = val

    @property
    def covariance(self):
        """Covariance of the gaussian distribution.

        Returns
        -------
        N x N numpy array.
        """
        return self._cov

    @covariance.setter
    def covariance(self, val:np.ndarray):
        self._cov = val

    @property
    def alpha(self):
        """Shape of the gaussian distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val

    @property
    def beta(self):
        """Rate of the gamma distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val

    @property
    def IWdof(self):
        """Degrees of freedom of the inverse Wishart distribution.

        Returns
        -------
        1 x 1 float.
        """
        return self._IWdof

    @IWdof.setter
    def IWdof(self, val:float):
        self._IWdof = val

    @property
    def IWshape(self):
        """Shape of the inverse Wishart distribution.

        Returns
        -------
        d x d numpy array.
        """
        return self._IWshape

    @IWshape.setter
    def IWshape(self, val:np.ndarray):
        self._IWshape = val 

    @property
    def d(self):
        return self._d
    
    def __eq__(self, obj):
        if isinstance(obj, GGIW):
            return (
                np.array_equal(self.alpha, obj.alpha) and
                np.array_equal(self.beta, obj.beta) and
                np.array_equal(self.mean, obj.mean) and
                np.array_equal(self.covariance, obj.covariance) and
                np.array_equal(self.IWdof, obj.IWdof) and
                np.array_equal(self.IWshape, obj.IWshape)
            )
        return False
    
    def __str__(self):
        # Build Gamma block (3 lines)
        gamma_lines = []
        gamma_lines.append("Gamma Distribution:")
        gamma_lines.append("  Gamma Shape (alpha): {:>12.4e}".format(self._alpha))
        gamma_lines.append("  Gamma Rate  (beta):  {:>12.4e}".format(self._beta))
        
        # Build Gaussian block with column headers for Mean and Covariance.
        gaussian_lines = []
        gaussian_lines.append("Gaussian Distribution:")
        gaussian_lines.append("  {:<30s}\t   {}".format("Mean", "Covariance"))
        
        dim = self._mean.size
        mean_parts = []
        cov_parts = []
        
        for i in range(dim):
            # Format mean element with appropriate bracket style.
            if dim == 1:
                mean_str = "[{:>12.4e}]".format(self._mean.ravel()[i])
            else:
                if i == 0:
                    mean_str = "\u2308{:>12.4e}\u2309".format(self._mean.ravel()[i])
                elif i == dim - 1:
                    mean_str = "\u230A{:>12.4e}\u230B".format(self._mean.ravel()[i])
                else:
                    mean_str = "|{:>12.4e}|".format(self._mean.ravel()[i])
            mean_parts.append(mean_str)
            
            # Format covariance row with matching Unicode bounds.
            cov_row = self._cov[i, :].tolist()
            if dim == 1:
                cov_str = "[{:>12.4e}]".format(cov_row[0])
            else:
                if i == 0:
                    cov_str = "\u2308" + ", ".join("{:>12.4e}".format(x) for x in cov_row) + "\u2309"
                elif i == dim - 1:
                    cov_str = "\u230A" + ", ".join("{:>12.4e}".format(x) for x in cov_row) + "\u230B"
                else:
                    cov_str = "|" + ", ".join("{:>12.4e}".format(x) for x in cov_row) + "|"
            cov_parts.append(cov_str)
        
        # Compute fixed column widths so that the covariance strings line up.
        mean_width = max(len(s) for s in mean_parts)
        cov_width = max(len(s) for s in cov_parts)
        
        for i in range(dim):
            # Added one extra tab (\t) between mean and covariance columns.
            line = "  {:<{mw}}\t   {:<{cw}}".format(mean_parts[i], cov_parts[i], mw=mean_width, cw=cov_width)
            gaussian_lines.append(line)
        
        # Build Inverse Wishart block.
        iw_lines = []
        iw_lines.append("Inverse Wishart Distribution:")
        iw_lines.append("  Degrees of Freedom: {:>12.4e}".format(self._IWdof))
        iw_lines.append("  Shape Matrix:")
        dim_iw = self._IWshape.shape[0]
        for i in range(dim_iw):
            row_str = "    [ " + " ".join("{:>12.4e}".format(x) for x in self._IWshape[i, :].tolist()) + " ]"
            iw_lines.append(row_str)
        
        # Combine the three blocks side by side.
        max_lines = max(len(gamma_lines), len(gaussian_lines), len(iw_lines))
        
        def pad_lines(lines, count):
            return lines + [""] * (count - len(lines))
        
        gamma_lines = pad_lines(gamma_lines, max_lines)
        gaussian_lines = pad_lines(gaussian_lines, max_lines)
        iw_lines = pad_lines(iw_lines, max_lines)
        
        gamma_width = max(len(line) for line in gamma_lines)
        gaussian_width = max(len(line) for line in gaussian_lines)
        
        combined_lines = []
        for g_line, ga_line, iw_line in zip(gamma_lines, gaussian_lines, iw_lines):
            combined_lines.append("{:<{gw}}   {:<{gw2}}   {}".format(
                g_line, ga_line, iw_line, gw=gamma_width, gw2=gaussian_width))
        
        return "\n".join(combined_lines) + "\n"
    
    def sample_measurements(self, xy_inds=[0,1], random_extent=False, random_state=None):
        """
        Simulate a set of measurements from this GGIW distribution:
          1) Number of points N ~ Poisson(alpha / beta),
          2) Target center ~ N(mean, covariance)   [if you want 
             to incorporate kinematic uncertainty]
          3) Extent ~ InverseWishart(IWshape, IWdof),
          4) Each measurement ~ N(center, extent).

        Returns
        -------
        measurements : (N x d) array of measurement vectors
                       (possibly zero-length if N=0).
        """
        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(random_state)

        # Mean number of measurements = alpha / beta
        lam = self._alpha / self._beta
        N = stats.poisson(lam).rvs(random_state=rng)

        # Target center is assumed to be the mean for this purpose, since sampling is for truth targets
        center = self._mean[xy_inds]

        # Sample the extent from an Inverse Wishart
        if random_extent:
            extent = stats.invwishart.rvs(df=self._IWdof, scale=self._IWshape, random_state=rng)
            # Sample each measurement from N(center, extent)
            if N > 0:
                measurements = rng.multivariate_normal(center.flatten(), extent, size=N)
            else:
                measurements = np.empty((0, self._d))
        else:
            extent = self._IWshape / (self.IWdof + self.d + 1)
            if N > 0:
                measurements = rng.multivariate_normal(center.flatten(), extent, size=N)
            else:
                measurements = np.empty((0, self._d))

        

        return measurements.T

    def plot_distribution(self, plt_inds=[0,1], ax=None, cov_std = 1.0, num_std=1.0, 
                          plot_covs=True, color = None, label = None, **kwargs):
        """
        Plot the GGIW in 2D:
          - draws an ellipse for the *mean* of the Inverse Wishart,
          - places a marker at the Gaussian mean.

        Parameters
        ----------
        ax : matplotlib Axes
            If None, uses current axes.
        num_std : float
            Number of "std devs" for the ellipse scaling.
        kwargs : dict
            Extra arguments passed to Ellipse (e.g., edgecolor, linestyle).
        """
        if ax is None:
            ax = plt.gca()

        if self.d != 2:
            raise ValueError("plot_distribution() only supports 2D for this example.")
        
        if label is None:
            label = ''

        # The ellipse center = Gaussian mean
        center = self._mean

         # Plot the center (Gaussian mean)
        p = ax.plot(center[plt_inds[0]], center[plt_inds[1]], 'o', color=color, label=label)

        #In case color is not set, this is how you get all cov and extension to be the same color
        if color is None:
            color = p[0].get_color()

        # Mean of Inverse Wishart(V, v) = V / (v - d - 1)  if v > d+1
        if self._IWdof <= self._d + 1:
            raise ValueError("Degrees of freedom must exceed d+1 for valid mean of IW.")
        mean_extent = self._IWshape / (self._IWdof - self._d - 1)

        # Decompose mean_extent to get ellipse axes
        eigvals, eigvecs = np.linalg.eigh(mean_extent)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Radii for ellipse = sqrt(eigenvalues) 
        r1, r2 = num_std * np.sqrt(eigvals)

        scale = np.sqrt(stats.chi2.ppf(0.99, df=2))

        r1 = r1 * scale
        r2 = r2 * scale

        # Orientation angle (the second eigenvector is the major axis if
        # it's the larger eigenvalue, but here we sorted them)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

        ellipse = Ellipse(
            xy=center[plt_inds],
            width=2*r2,    # total width
            height=2*r1,   # total height
            angle=angle,
            fill=False,
            color = color
        )
        ax.add_patch(ellipse)

        if plot_covs:
            # Extract the 2x2 sub-block of self.cov to match plt_inds (x,y, for example)
            cov_2d = self._cov[np.ix_(plt_inds, plt_inds)]

            eigvals_c, eigvecs_c = np.linalg.eigh(cov_2d)
            order_c = np.argsort(eigvals_c)
            eigvals_c = eigvals_c[order_c]
            eigvecs_c = eigvecs_c[:, order_c]

            r1_c, r2_c = cov_std * np.sqrt(eigvals_c)
            angle_c = np.degrees(np.arctan2(eigvecs_c[1, 1], eigvecs_c[0, 1]))

            cov_ellipse = Ellipse(
                xy=center[plt_inds],
                width=2*r2_c,
                height=2*r1_c,
                angle=angle_c,
                fill=False,
                linestyle='--',   # maybe dashed to distinguish from extent
                linewidth=3,
                color = color
            )
            cov_ellipse.set_alpha(0.5)
            ax.add_patch(cov_ellipse)

       
        ax.set_aspect('equal', 'box') 

    def plot_confidence_extents(self, h=0.95, plt_inds=[0, 1], ax=None, plot_mean=True, **kwargs):
        """
        Plot two dashed ellipses for the Inverse Wishart (IW) 'extent' using
        the given confidence values h_min, h_max. 

        Parameters
        ----------
        h_min : float
            Lower quantile (e.g., 0.05).
        h_max : float
            Upper quantile (e.g., 0.95).
        plt_inds : list of int
            Indices for which 2D plane to visualize (default [0,1]).
        ax : matplotlib.axes.Axes
            Axes object. If None, uses current axes.
        kwargs : dict
            Extra properties passed to the Ellipse constructor
            (e.g., edgecolor, alpha, etc.).
        """
        if ax is None:
            ax = plt.gca()

        if self.d != 2:
            raise ValueError("plot_confidence_extents() only supports 2D for this example.")
        
        center = self._mean

        if self._IWdof <= self._d + 1:
            raise ValueError("Degrees of freedom must exceed d+1 for valid IW mean.")

        mean_extent = self._IWshape / (self._IWdof - self._d - 1)

        eigvals, eigvecs = np.linalg.eigh(mean_extent)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Orientation angle 
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

        # scale by sqrt(chi2.ppf(h, df=2)) since d_k|k follows a chi2 distribution
        scale = np.sqrt(stats.chi2.ppf(h, df=2))
        r1 = scale * np.sqrt(eigvals[0]) 
        r2 = scale * np.sqrt(eigvals[1]) 
        confidence_e = Ellipse(
            xy=center[plt_inds],
            width=2*r2,
            height=2*r1,
            angle=angle,
            fill=False,
            linestyle='--', 
            **kwargs
        )

        ax.add_patch(confidence_e)

        if plot_mean:
            r1 = np.sqrt(eigvals[0]) 
            r2 = np.sqrt(eigvals[1]) 
            e = Ellipse(
                xy=center[plt_inds],
                width=2*r2,
                height=2*r1,
                angle=angle,
                fill=False, 
                **kwargs
            ) 
            ax.add_patch(e)

        ax.set_aspect('equal', 'box')

class GGIWMixture(BaseMixtureModel):
    """Gamma Gaussian Inverse Wishart Mixture object."""
    
    def __init__(self, alphas=None, betas=None, means=None, covariances =None, IWdofs =None, IWshapes = None,**kwargs):
        """Initialize a Mixture object. """

        if means is not None and covariances is not None and alphas is not None and betas is not None and IWdofs is not None and IWshapes is not None:
            kwargs["distributions"] = [
                GGIW(alpha=a, beta=b, mean=m, covariance=c, IWdof=v, IWshape=V) for a, b, m, c, v, V in zip(alphas, betas, means, covariances, IWdofs, IWshapes)
            ] 
            #kwargs["weights"] = [1 / len(alphas) for _ in range(len(alphas))]
        super().__init__(**kwargs)

    def __getitem__(self,idx):
        return self._distributions[idx]

    @property
    def means(self):
        """List of Gaussian means for the GGIW components (each is an N x 1 numpy array). Recommended to be read only."""
        return _DistListWrapper(self._distributions, "_mean")

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
        return _DistListWrapper(self._distributions, "_cov")

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
        return _DistListWrapper(self._distributions, "_alpha")

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
        return _DistListWrapper(self._distributions, "_beta")

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
        return _DistListWrapper(self._distributions, "_IWdof")

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
        return _DistListWrapper(self._distributions, "_IWshape")

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
    
    def add_components(self, alphas:list=None, betas:list=None, 
                       means:list=None, covariances:list=None, 
                       IWdofs:list=None, IWshapes:list=None, weights=None, ggiw:GGIW=None):
        """Add GGGIW distributions to the mixture."""
        if ggiw is None:
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
        else:
            # If we input the entire GGIW distribution and weight
            self._distributions.extend([ggiw])
            self.weights.extend([weights])


    def __str__(self):
        s = ""
        for ii in range(len(self._distributions)):
            s += f"Term {ii+1}: (weight = {self.weights[ii]})\n"
            s += str(self._distributions[ii])
        return s
    
    def plot_confidence_extents(self, h=0.95, plt_inds=[0, 1], ax=None, plot_mean=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        for ii in range(len(self._distributions)):
            cur_dist = self._distributions[ii]
            cur_dist.plot_confidence_extents(h=h,plt_inds=plt_inds,ax=ax,plot_mean=plot_mean,**kwargs)

    def plot_distributions(self,plt_inds=[0,1], ax=None, cov_std=1.0, num_std=2.0, plot_covs=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        for ii in range(len(self._distributions)):
            cur_dist = self._distributions[ii]
            cur_dist.plot_distribution(plt_inds, ax, cov_std, num_std, plot_covs, **kwargs)