#ARD EM
ARD (Automatic Relevance Determination) EM implementation on Python.
It's EM algorithm with automatic determination of number of components. It can also be used for clusterization with unknown number of components.

[ArdGaussianMixture](gaussian_ard_mixture.py) class has the same interface as SkLearn's [GaussianMixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) with 3 additional parameters:
```python
init_components="sqrt" # Initial number of components. sqrt(N) if "sqrt"
alpha_bound=1e3 # Drop all components with weight_reg (alpha) > alpha_bound
weight_bound=1e-3 # Drop all components with weight < weight_bound
```

## Example
```python
from gaussian_ard_mixture import ArdGaussianMixture
gmm = ArdGaussianMixture()
gmm = gmm.fit(X)
print('Bayesian information criterion: ', gmm.bic(X))
best_n_components = gmm.n_components
gmm.predict(X)
```

## Links
[Original paper](http://www.machinelearning.ru/wiki/images/d/dc/Vetrov-ArdEm-JVMMF-2009.pdf)