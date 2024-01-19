install.packages("https://okuno.net/R-packages/sgdpd_1.0.0.zip", repos=NULL, type="win.binary")
require(sgdpd)
require(fGarch)

## dataset generation (skew-normal)
Z = as.matrix(append(rsnorm(n=700, mean=-2, sd=1, xi=4), rsnorm(n=300, mean=3, sd=0.5, xi=1/3)), 1000, 1)

## skew normal mixture density
f_snm <- function(z, theta){
  alpha = sigmoid(theta[7])
  alpha * dsnorm(x=z, mean=theta[1], sd=theta[2], xi=theta[3]) + (1-alpha) * dsnorm(x=z, mean=theta[4], sd=theta[5], xi=theta[6])
}

## parameter estimation
par_snm= sgdpd(f=f_snm, Z=Z, lr=0.1, theta0=c(-1,1,1,1,1,1,0), positives=c(2,3,5,6), exponent=0.2)

## plot
xl = range(Z); xx = matrix(seq(xl[1], xl[2], length.out=100), 100, 1)
hist(Z, breaks=100, xlim=xl); par(new=T); 
plot(xx, apply_f(f_snm, xx, par_snm$theta), xlim=xl, type="l", col="blue", lwd=2,
     main=" ", xlab=" ", ylab=" ", xaxt="n", yaxt="n")