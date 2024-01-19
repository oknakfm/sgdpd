\name{sgdpd}
\alias{sgdpd}
\title{stochastic gradient descent for density power divergence minimization}
\description{this function minimizes density power divergence by stochastic gradient descent}
\usage{
sgdpd(f, Z, theta0, lr, positives, conditions, 
exponent, N, M, showProgress, h, log_interval)
}
\arguments{
    \item{f}{(necessary) non-negative parametric model f=f(z,theta) to be optimized.}
    \item{Z}{(necessary) n*d design matrix, where n represents sample size and d represents the dimension.}
    \item{theta0}{(necessary) initial parameter.}
    \item{lr}{(necessary) learning rate schedule. you can input vector (for schedule), or a positive real value (then the subsequent schedule is automatically generated).}
    \item{positives}{(optional) parameter index, whose estimated value should be positive (e.g., standard deviation of normal distriubtion).}
    \item{conditions}{(optional) if you consider regression problem, specify the outcome index in the design matrix Z.}
    \item{exponent}{(optional) exponent of density power divergence (default: 0.1).}
    \item{N}{(optional) sample size for the 1st term.}
    \item{M}{(optional) sample size for the 2nd term.}
    \item{showProgress}{(optional) if TRUE, the optimization progress is shown (default: TRUE).}
    \item{h}{(optional) bandwidth for numerical differentiation.}
    \item{log_interval}{(optional) nterval to trace the parameter update. if 0, no trace.}
}
\details{
theta0 is the initial parameter, lr is the learning rate, exponent is the exponent of the density power divergence. (exponent=0 represents the Kullback-Leibler divergence)
}
\value{
log represents the traced parameter (via the stochastic optimization) with the interval log_interval, and theta represents the estimated parameter.
}
\examples{
Z = matrix(rnorm(n=(n<-1000), mean=2, sd=2), n, 1)
f <- function(z, theta) dnorm(x=z, mean=theta[1], sd=theta[2])
sgdpd(f=f, Z=Z, theta0=c(0,1), lr=0.1, exponent=0.1)
}
