<p align="center">
<img src="./readme/sgdpd.png" width=400></img>
</p>

## Overview
This repository offers a user-friendly `R` package of an optimizer of density power divergence (Basu et al. <a href="https://doi.org/10.1093/biomet/85.3.549">Biometrika 1998</a>) :
```math
L_{\alpha}(\theta)
=
-\frac{1}{\alpha} \frac{1}{n} \sum_{i=1}^{n} f_{\theta}(z_i)^{\alpha} + \frac{1}{1+\alpha}\int f_{\theta}(z)^{1+\alpha} dz.
```
This optimizer uses a stochastic gradient descent (proposed by Okuno <a href="https://arxiv.org/abs/2307.05251">arXiv:2307.05251</a>), and 
it needs minimal effort for users to obtain the optimal parameter. 

## Install
```R:install
install.packages("https://okuno.net/R-packages/sgdpd_1.0.0.zip", repos=NULL, type="win.binary"))
```

## Quickstart
To demonstrate our optimizer, we here conduct a univariate skew-normal density estimation. Firstly, we specify the skew-normal density function parameterized by theta (where the `dsnorm` function is called from `fGarch` package):
```R:f
f <- function(z, theta) dsnorm(x=z, mean=theta[1], sd=theta[2], xi=theta[3])
```
With the design matrix $`Z=(z_1^{\top},z_2^{\top},\ldots,z_n^{\top})^{\top}`$, learning rate schedule and a initial parameter, we can obtain the optimal parameter by simply calling our optimizer (where the default exponent is 0.1) as:
```R:sgd_dpd
sgdpd(f=f, Z=Z, lr=0.1, theta0=c(0,1,1))
```
No further operation is needed!

## Contact info.
- Akifumi Okuno (ISM and RIKEN AIP) https://okuno.net
