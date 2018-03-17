# ARD EM
ARD (Automatic Relevance Determination) EM implementation on Python.
It's EM algorithm with automatic determination of number of components. It's powerful and fast algorithm for gaussian mixture learning and clustering with unknown number of components.

The implemented [ArdGaussianMixture](gaussian_ard_mixture.py) class has the same interface as SkLearn's [GaussianMixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) one, but with 3 additional parameters:
```python
init_components="sqrt" # Initial number of components. sqrt(N) if "sqrt"
alpha_bound=1e3 # Drop all components with weight_reg (alpha) > alpha_bound
weight_bound=1e-3 # Drop all components with weight < weight_bound
```
and without **n_components** one.

## Example
```python
from gaussian_ard_mixture import ArdGaussianMixture
gmm = ArdGaussianMixture()
gmm = gmm.fit(X)
print('Bayesian information criterion: ', gmm.bic(X))
best_n_components = gmm.n_components
print('Best number of components: ', best_n_components)
gmm.predict(X)
```

## Links
[Original paper](http://www.machinelearning.ru/wiki/images/d/dc/Vetrov-ArdEm-JVMMF-2009.pdf)

## Author
Artem Ryzhikov, LAMBDA laboratory, Highest School of Economics, Yandex School of Data Analysis
**E-mail:** artemryzhikoff@yandex.ru
**Linkedin:** https://www.linkedin.com/in/artem-ryzhikov-2b6308103/
**HSE profile:** https://www.hse.ru/org/persons/190912317
