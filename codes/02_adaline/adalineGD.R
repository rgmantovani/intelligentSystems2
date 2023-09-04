# -----------------------------------------------------------------
# -----------------------------------------------------------------

# ADALINE Regression (predicting real values)

# during training:
# phi(v) = v

# after training
# phi(v) = ifelse(v >=0, +1, -1)

# η > 0 is the learning rate, typically 0 < η << 0.1

# batch learning: updates weights after evaluating all the examples

# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineGD.train = function(dataset, n.iter = 100, lrn.rate = 0.0001,
  classification = TRUE) {

  # we are dealing with extended input and weight vectors (bias)
  X = cbind(1, dataset)

  class.id = ncol(X)
  y = as.numeric(X[,class.id])
  X = as.matrix(X[,-class.id])

  # generating random weights [-0.5, 0.5]
  w = matrix(data = runif(n = ncol(dataset), min = -0.5, max = 0.5))

  # initialize vector to keep track of error and misc function per epoch
  error = rep(0, n.iter)
  misc  = rep(0, n.iter)

  # loop over the number of epochs
  for (n in 1:n.iter) {

    cat("* Epoch:", n)

    # linear sum of values
    Y = lapply(1:nrow(X), function(i) {
      v = sum(w * X[i,])
      return(v)
    })
    ypred  = unlist(Y)

    # error in the current epoch and the cost function
    if(classification) {
      # quantitizer
      ypred[ypred >= 0] = +1
      ypred[ypred < 0]  = -1
    }

    misc[n]  = length(which(y != ypred))
    error[n] = sum((y - X %*% w)^2)/2
    cat(" - error: ", error[n], " - misc: ", misc[n], "\n")

    # update weight according to gradient descent
    w = w + lrn.rate *t(X) %*% (y - X %*% w)
  }

  # returning the model
  model = list(epochs = n.iter, misc = misc, lrn.rate = lrn.rate,
    error = error, weights = w)
  
  return(model)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineGD.predict = function(example, weights) {

  example = c(1, example)
  z = sum(example * obj$weights)
  y = ifelse(z >= 0, +1, -1)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineGD.regress = function(example, weights) {

  example = c(1, example)
  z = sum(example * obj$weights)
  return(z)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
