from sklearn.mixture.gaussian_mixture import *
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_covariances_diag, \
  _estimate_gaussian_covariances_full, _estimate_gaussian_covariances_spherical, \
  _estimate_gaussian_covariances_tied, _compute_precision_cholesky
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.mixture.base import _check_X
import warnings
from numpy.linalg import LinAlgError



def _estimate_ard_parameters(X, w_old, reg_weights, resp, reg_covar, covariance_type):
  """Estimate the Gaussian distribution parameters.

  Parameters
  ----------
  X : array-like, shape (n_samples, n_features)
      The input data array.

  w_old : array-like, shape (n_components,)
      The old weights of components ("w_old" from paper)
      
  reg_weights : array-like, shape (n_components,)
      The weights regularization ("alpha" from paper)
  
  resp : array-like, shape (n_samples, n_components)
      The responsibilities for each data sample in X.

  reg_covar : float
      The regularization added to the diagonal of the covariance matrices.

  covariance_type : {'full', 'tied', 'diag', 'spherical'}
      The type of precision matrices.

  Returns
  -------
  nk : array-like, shape (n_components,)
      The numbers of data samples in the current components.

  means : array-like, shape (n_components, n_features)
      The centers of the current components.

  covariances : array-like
      The covariance matrix of the current components.
      The shape depends of the covariance_type.
  """
  n_samples, _ = X.shape
  nk_reg = (w_old ** 2) * reg_weights
  nk = (resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps - nk_reg) * n_samples / (n_samples - nk_reg.sum())
  means = np.dot(resp.T, X) / nk[:, np.newaxis]
  covariances = {"full": _estimate_gaussian_covariances_full,
                 "tied": _estimate_gaussian_covariances_tied,
                 "diag": _estimate_gaussian_covariances_diag,
                 "spherical": _estimate_gaussian_covariances_spherical
                 }[covariance_type](resp, X, nk, means, reg_covar)
  return nk, means, covariances


class ArdGaussianMixture(GaussianMixture):
  
  def __init__(self, init_components='sqrt', alpha_bound=1e3, weight_bound=1e-3,
               covariance_type='full', tol=1e-3,
               reg_covar=1e-6, max_iter=100, n_init=3, init_params='kmeans',
               weights_init=None, means_init=None, precisions_init=None,
               random_state=None, warm_start=False,
               verbose=0, verbose_interval=10):
    """
    
    :param initial_components: str or number. Initial number of components. sqrt(N) if "sqrt"
    :param alpha_bound: float. Drop all components with weight_reg (alpha) > alpha_bound
    :param weight_bound: float. Drop all components with weight < weight_bound
    """
    super(ArdGaussianMixture, self).__init__(1, covariance_type, tol,
               reg_covar, max_iter, n_init, init_params,
               weights_init, means_init, precisions_init,
               random_state, warm_start,
               verbose, verbose_interval)
    self.init_components = init_components
    self.alpha_bound = alpha_bound
    self.weight_bound = weight_bound
  
  def _initialize_parameters(self, X, random_state):
    n_samples, _ = X.shape
  
    if self.init_components == 'sqrt':
      n_components = int(np.sqrt(n_samples))
    else:
      n_components = self.init_components
    self.n_components = n_components
  
    initial_clf = GaussianMixture(self.n_components, self.covariance_type, self.tol,
                                  self.reg_covar, self.max_iter, self.n_init,
                                  self.init_params, self.weights_init, self.means_init,
                                  self.precisions_init, self.random_state, self.warm_start,
                                  self.verbose, self.verbose_interval)
  
    initial_clf.fit(X)
  
    self.weights_ = initial_clf.weights_
    self.means_ = initial_clf.means_
    self.covariances_ = initial_clf.covariances_
    self.precisions_cholesky_ = initial_clf.precisions_cholesky_
    self.reg_weights_ = np.ones((self.n_components,))
  
  
  def _m_step(self, X, resp):
      """M step.
    
              Parameters
              ----------
              X : array-like, shape (n_samples, n_features)
    
              resp : array-like, shape (n_samples, n_components)
                  Posterior probabilities (or responsibilities) of
                  the point of each sample in X.
              """
      n_samples, _ = X.shape
      self.weights_, self.means_, self.covariances_ = (
        _estimate_ard_parameters(X, self.weights_, self.reg_weights_, resp, self.reg_covar,
                                 self.covariance_type))
      self.weights_ /= n_samples
      self.precisions_cholesky_ = _compute_precision_cholesky(
        self.covariances_, self.covariance_type)

  def _get_parameters(self):
    return (self.weights_, self.means_, self.covariances_,
            self.precisions_cholesky_, self.reg_weights_)

  def _set_parameters(self, params):
    (self.weights_, self.means_, self.covariances_,
     self.precisions_cholesky_, self.reg_weights_) = params
  
    self.n_components = self.weights_.shape[0]
  
    # Attributes computation
    _, n_features = self.means_.shape
  
    if self.covariance_type == 'full':
      self.precisions_ = np.empty(self.precisions_cholesky_.shape)
      for k, prec_chol in enumerate(self.precisions_cholesky_):
        self.precisions_[k] = np.dot(prec_chol, prec_chol.T)
  
    elif self.covariance_type == 'tied':
      self.precisions_ = np.dot(self.precisions_cholesky_,
                                self.precisions_cholesky_.T)
    else:
      self.precisions_ = self.precisions_cholesky_ ** 2

  def fit(self, X, y=None): # replace to iterative formula using super ._e_step() and overrided ._m_step() 3.5
    X = _check_X(X, self.n_components)
    self._check_initial_parameters(X)
  
    # if we enable warm_start, we will have a unique initialisation
    do_init = not (self.warm_start and hasattr(self, 'converged_'))
    n_init = self.n_init if do_init else 1
  
    max_lower_bound = -np.infty
    self.converged_ = False
  
    random_state = check_random_state(self.random_state)
  
    n_samples, _ = X.shape
    for init in range(n_init):
      self._print_verbose_msg_init_beg(init)
    
      if do_init:
        self._initialize_parameters(X, random_state)
        self.lower_bound_ = -np.infty
    
      for n_iter in range(self.max_iter):
        # ARD EM
        prev_lower_bound = self.lower_bound_
        log_prob_norm, log_resp = self._e_step(X)
        resp = np.exp(log_resp)
        self._m_step(X, resp)
        
        # resp : array-like, shape (n_samples, n_components)
        # The responsibilities for each data sample in X.
        
        # update covariance
        F = (1./(np.diag(np.dot(resp, self.weights_)) + 1e-8)) ** 2 # array-like, shape (n_samples, n_samples)
        A = np.diag(self.reg_weights_) # array-like, shape (n_components, n_components)
        H = np.dot(np.dot(resp.T, F), resp) + A
        S = np.vstack([np.diag(np.ones(self.n_components)), np.ones(self.n_components)])
        try:
          Sigma = np.linalg.inv(np.dot(np.dot(S, H), S.T))
        except LinAlgError:
          break
        idx = np.arange(self.n_components - 1)
        
        self.reg_weights_ = np.hstack([
          (1 - self.reg_weights_[idx] * Sigma[idx, idx]) / self.weights_[idx] ** 2,
          (1 - self.reg_weights_[-1] * Sigma.sum()) / self.weights_[-1] ** 2
        ])
        
        # drop extra components
        idx = np.argwhere(np.logical_and(self.reg_weights_ < self.alpha_bound, self.weights_ > self.weight_bound)).squeeze()
        self.weights_ = self.weights_[idx]
        self.reg_weights_ = self.reg_weights_[idx]
        self.means_ = self.means_[idx, :]
        if self.covariance_type != 'tied':
          self.covariances_ = self.covariances_[idx]
          self.precisions_cholesky_ = self.precisions_cholesky_[idx]
        self.n_components = idx.shape[0]
        
        # early stop cryterium
        self.lower_bound_ = self._compute_lower_bound(
          log_resp, log_prob_norm)
        change = self.lower_bound_ - prev_lower_bound
        self._print_verbose_msg_iter_end(n_iter, change)
        if abs(change) < self.tol:
          self.converged_ = True
          break
    
      self._print_verbose_msg_init_end(self.lower_bound_)
    
      if self.lower_bound_ > max_lower_bound:
        max_lower_bound = self.lower_bound_
        best_params = self._get_parameters()
        best_n_iter = n_iter
  
    if not self.converged_:
      warnings.warn('Initialization %d did not converge. '
                    'Try different init parameters, '
                    'or increase max_iter, tol '
                    'or check for degenerate data.'
                    % (init + 1), ConvergenceWarning)
  
    self._set_parameters(best_params)
    self.n_iter_ = best_n_iter
  
    return self