<p align="center">
<img src="./readme/sgdpd.png" width=400></img>
</p>

# Overview
This repository offers a user-friendly `R` package for an optimizer minimizing the density power divergence proposed by <a href="https://doi.org/10.1093/biomet/85.3.549">Basu et al. (1998)</a> :
```math
L_{\alpha}(\theta)
=
-\frac{1}{\alpha} \frac{1}{n} \sum_{i=1}^{n} f_{\theta}(z_i)^{\alpha} + \frac{1}{1+\alpha}\int f_{\theta}(z)^{1+\alpha} dz.
```
This optimizer needs minimal effort for users to obtain the optimal parameter, to estimate general parametric models. 

# Quickstart
## Install
Please enter and execute the following command to install our `sgdpd` package.
```R:install
install.packages("https://okuno.net/R-packages/sgdpd_1.0.0.zip", repos=NULL, type="win.binary")
```

## Example
In order to showcase the capabilities of our optimizer, we perform a univariate skew-normal density estimation. Initially, we define the skew-normal density function, parameterized by theta. This is achieved using the dsnorm function, which is sourced from the fGarch package:
```R:f
f <- function(z, theta) dsnorm(x=z, mean=theta[1], sd=theta[2], xi=theta[3])
```
Using the $`n \times d`$ design matrix `Z`, along with a specified learning rate `lr`, an initial parameter `theta0`, and an exponent parameter `exponent`, we can efficiently compute the optimal parameter as follows: 
```R:sgd_dpd
sgdpd(f=f, Z=Z, lr=0.1, theta0=c(0,1,1), exponent=0.2)
```
No further operation is needed! Please also try <a href="https://github.com/oknakfm/sgdpd/blob/main/demo.R">our demo</a> for skew-normal mixture density estimation. 

## Contact info.
- Akifumi Okuno (ISM and RIKEN AIP) https://okuno.net
