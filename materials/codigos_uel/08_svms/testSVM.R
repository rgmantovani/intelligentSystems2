# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

rm(list = ls())
library('e1071')

x1s = c(.5,1,1,2,3,3.5,     1,3.5,4,5,5.5,6)
x2s = c(3.5,1,2.5,2,1,1.2,  5.8,3,4,5,4,1)
ys  = c(rep(+1,6),          rep(-1,6))
my.data = data.frame(x1=x1s, x2=x2s, type=as.factor(ys))
my.data

# seeing our data
plot(my.data[,-3], col=(ys+3)/2, pch=19, xlim=c(-1,6), ylim=c(-1,6))

# training a svm
svm.model = e1071::svm(type ~ ., data = my.data,
  type='C-classification',
  kernel='linear', scale = FALSE)

svm.model

# ploting data and support vectors
plot(my.data[,-3],col=(ys+3)/2, pch=19, xlim=c(-1,6), ylim=c(-1,6))
points(my.data[svm.model$index,c(1,2)],col="blue",cex=2)

# getting hyperplane and marging
w = t(svm.model$coefs) %*% svm.model$SV
b = -svm.model$rho
p = svm.model$SV

# plotting marging
abline(a=-b/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=1)
abline(a=(-b-1)/w[1,2], b=-w[1,1]/w[1,2], col="red", lty=3)
abline(a=(-b+1)/w[1,2], b=-w[1,1]/w[1,2], col="red", lty=3)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
