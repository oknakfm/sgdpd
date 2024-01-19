\name{apply_f}
\alias{apply_f}
\title{apply_f}
\description{this function applies the function f to all the entries in the design matrix Z}
\usage{
apply_f(f, Z, theta)
}
\arguments{
    \item{f}{non-negative parametric model f=f(z,theta) to be optimized.}
    \item{Z}{n*d design matrix, where n represents sample size and d represents the dimension.}
    \item{theta}{parameter.}
}
\details{
function f applied to all the entries in the design matrix Z.
}
\value{
real-valued vector.
}
\examples{
Z = matrix(rnorm(n=(n<-10), mean=2, sd=2), n, 1)
f <- function(z, theta) dnorm(x=z, mean=theta[1], sd=theta[2])
apply_f(f, Z, theta=c(0,1))
}




