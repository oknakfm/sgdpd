\name{lr_scheduler}
\alias{lr_scheduler}
\title{learning rate scheduler}
\description{this function provides a schedule of the learning rate}
\usage{
lr_scheduler(n_itr, lr0, gamma, decay_period, cyclic_period)
}
\arguments{
    \item{n_itr}{number of iteration}
    \item{lr0}{initial learning rate}
    \item{gamma}{decay rate}
    \item{decay_period}{decay period}
    \item{cyclic_period}{cyclic period}
}
\details{
the learning rate schedule is started from lr0, and the rate is multiplied by gamma for each decay_period.
for each of cyclic_period iteration, the learning rate is reset to lr0, and multiply the reset value by gamma (itr/cyclic_period) times.
}
\value{
learning rate schedule (vector of length n_itr)
}
\examples{
lr_scheduler(n_itr=3000, lr0=0.1, gamma=0.8, decay_period=50, cyclic_period=200)
}




