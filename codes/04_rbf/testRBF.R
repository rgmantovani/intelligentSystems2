# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("../RBF.R")

set.seed(42)

# -----------------------------------------------------------------
# generate artificial data
# -----------------------------------------------------------------

target = function(x1, x2) {
  y = 2*(x2 - x1 + .25*sin(pi*x1) >= 0)-1
  return(y)
}

# generating dataset
N = 100
X = data.frame(x1=runif(N, min=-1, max=1), x2=runif(N, min=-1, max=1))
Y = target(x1 = X$x1, x2 = X$x2)
train.data = cbind(X, Y)

# view data
plot(train.data$x1, train.data$x2, col=train.data$Y+3)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# testing data
N.test = 200
X.out = data.frame(x1=runif(N.test, min=-1, max=1),
  x2 = runif(N.test, min=-1, max=1))
Y.out = target(X.out$x1, X.out$x2)
test.data = cbind(X.out, Y.out)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# train RBF
rbf.model = rbf.train(dataset = train.data, K = 10, gamma = 1)

# predit data
rbf.pred = rbf.predict(model = rbf.model, X = test.data[, -ncol(test.data)],
  classification = TRUE)

tab = table(Y.out, rbf.pred)
acc = sum(diag(tab))/N.test
print(acc)

plot(test.data$x1, test.data$x2, col=Y.out+3, pch=0)
points(X.out$x1, X.out$x2, col=rbf.pred+3, pch=3)

# draw the model centers
points(rbf.model$centers, col="black", pch=19)
legend("topleft",c("true value","predicted"),pch=c(0,3),bg="white")

# -----------------------------------------------------------------
# -----------------------------------------------------------------
