# ARD EM
ARD (Automatic Relevance Determination) EM implementation on Python.
The classical EM-algorithm for reconstructing a mixture of normal distributions does not allow to determine the amount of components of the mixture. The ARD EM implementation suggests algorithm for automatically determining the number of components ARD EM, based on the method of relevant vectors. The idea of the algorithm is to use at the initial stage of a knowingly excessive amount of the components of the mixture with further determination of the relevant components by maximizing
validity. Experiments on model problems show that the number of found clusters either coincides with the true one, or slightly
excels him. In addition, clustering with ARD EM is closer to the true than the analogs based on sliding control and
character of the minimum description length. It's EM algorithm with automatic determination of number of components. It's powerful and fast algorithm for gaussian mixture learning and clustering with unknown number of components.

# Implementation
The implemented [GaussianMixtureARD](ard_em.py) class has the same interface as SkLearn's [GaussianMixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) one, but with 3 additional parameters:
```python
init_components="sqrt" # Initial number of components. sqrt(N) if "sqrt"
alpha_bound=1e3 # Drop all components with weight_reg (alpha) > alpha_bound
weight_bound=1e-3 # Drop all components with weight < weight_bound
```
and without **n_components** one.

# Installation
```
pip install git+https://github.com/Leensman/ard-em.git
```

## Example
```python
from ard_em import GaussianMixtureARD
gmm = GaussianMixtureARD()
gmm = gmm.fit(X)
print('Bayesian information criterion: ', gmm.bic(X))
best_n_components = gmm.n_components
print('Best number of components: ', best_n_components)
gmm.predict(X)
```
For more examples go to [GaussianMixture.ipynb](https://github.com/Leensman/ard-em/blob/master/ard-em/examples/Gaussian%20mixture.ipynb)

## Links
[Original paper](http://www.machinelearning.ru/wiki/images/d/dc/Vetrov-ArdEm-JVMMF-2009.pdf)

## Author
Artem Ryzhikov, LAMBDA laboratory, Higher School of Economics, Yandex School of Data Analysis

**E-mail:** artemryzhikoff@yandex.ru

**Linkedin:** https://www.linkedin.com/in/artem-ryzhikov-2b6308103/

**HSE profile:** https://www.hse.ru/org/persons/190912317
