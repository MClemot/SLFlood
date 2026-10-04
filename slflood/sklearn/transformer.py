import slflood

from multipers.filtrations.density import KDE, DTM
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch

class SublevelFloodBifiltration(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer computing the sublevel Flood bifiltration of point clouds.
    For each input point cloud, a co-density function is estimated via kernel density or distance-to-measure.

    Parameters
    ----------
    n_lms : int
        Number of landmarks to sample (via farthest point sampling).

    kde_bandwidth : float, optional
        Bandwidth of the Gaussian kernel used to compute the density functions.
        Mutually exclusive with ``dtm_mass``.

    dtm_mass : float, optional
        Mass parameter in (0, 1] used to compute the Distance-To-Measure (DTM) functions.
        Mutually exclusive with ``kde_bandwidth``.

    log_density : bool, default=False
        If True and ``kde_bandwidth`` is set, uses the log-density instead of the density.

    normalize : bool, default=True
        Whether to normalize the density functions.

    points_per_edge : int, default=20
        Resolution of the sampling grid used on each simplex.

    exact : bool, default=False
        Whether to compute the exact (unmasked) or approximate (masked, faster) sublevel Flood bifiltration.

    device : {'auto', 'cpu', 'cuda'}, default='auto'
        Device used for computation. When 'auto', detects whether CUDA is available.

    use_triton : bool, default=True
        Whether to use Triton kernels when available.

    n_jobs : int, default=-1
        Currently unused.
    """

    def __init__(self, n_lms, kde_bandwidth=None, dtm_mass=None, log_density=False, normalize=True, points_per_edge=20, exact=False, device='auto', use_triton=True, n_jobs=-1):
        self.n_lms = n_lms
        self.kde_bandwidth = kde_bandwidth
        self.dtm_mass = dtm_mass
        self.log_density = log_density
        self.normalize = normalize
        self.points_per_edge = points_per_edge
        self.exact = exact
        self.device = device
        self.use_triton = use_triton
        self.n_jobs = n_jobs

        assert (self.kde_bandwidth is None or self.dtm_mass is None)
        assert (self.kde_bandwidth is not None or self.dtm_mass is not None)

    def __get_device(self):
        if self.device not in ['cpu', 'cuda', 'auto']:
            raise ValueError(f"device must be 'cpu', 'cuda' or 'auto', got {self.device!r}")
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available")
        return self.device

    def fit(self, X, y=None):
        return self

    def __density(self, X):
        functions = []
        for pts in X:
            if self.kde_bandwidth is not None:
                function = -KDE(bandwidth=self.kde_bandwidth, kernel="gaussian", return_log=self.log_density).fit(pts).score_samples(pts)
            else:
                function = DTM(masses=[self.dtm_mass]).fit(pts).score_samples(pts)[0]
            if self.normalize:
                function -= function.min()
                function /= function.max()
            functions.append(function)
        return functions

    def __transform(self, x, f, device):
        x = torch.as_tensor(x, device=device)
        f = torch.as_tensor(f, device=device)
        return slflood.slflood_bifiltration(x, self.n_lms, f,
                                            points_per_edge=self.points_per_edge,
                                            exact=self.exact,
                                            use_triton=self.use_triton)

    def transform(self, X):
        device = self.__get_device()
        F = self.__density(X)

        return [self.__transform(x, f, device) for (x,f) in zip(X, F)]