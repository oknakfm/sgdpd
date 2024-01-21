install.packages("https://okuno.net/R-packages/sgdpd_1.0.0.zip", 
  repos=NULL, type="win.binary")

require("sgdpd")

## dataset generation (normal random)
Z = as.matrix(rnorm(n=1000, mean=3, sd=3), 1000, 1)

## model specification (normal density)
f_n <- function(z, theta) dnorm(x=z, mean=theta[1], sd=theta[2])

## initialization
theta0 <- c(0,1)

## parameter estimation
par_n = sgdpd(f=f_n, Z=Z, lr=0.1, theta0=theta0, positives=c(2), exponent=0.2)

## estimated parameter
par_n$theta
