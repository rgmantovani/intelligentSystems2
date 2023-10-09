# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineSGD.train = function(dataset, n.iter = 100,
  lrn.rate = 0.0001, classification = TRUE) {

  # we are dealing with extended input and weight vectors
  X = cbind(1, dataset)
  class.id = ncol(X)
  y = as.numeric(X[,class.id])
  X = as.matrix(X[,-class.id])

  # TODO: generate random weights [-0.5, 0.5]
  w = as.matrix(rep(0, dim(X)[2]))

  # initialize vector to keep track of cost and error function per epoch
  cost  = rep(0, n.iter)
  error = rep(0, n.iter)

  # loop through each epoch
  for(n in 1:n.iter) {

    cat("* Epoch:", n)
    # loop through each data point
    for(i in sample(1:dim(X)[1], dim(X)[1], replace = FALSE)) {

      # keep track of incorrect predictions
      z = sum(w * X[i, ])

      if(classification) {
        #quantizer
        ypred = ifelse(z < 0.0, -1, +1)
      } else {
        ypred = z
      }

      # count the number of misclassifications
      if(ypred != y[i]) {
        error[n] = error[n] + 1
      }

      # update weights
      w = w + lrn.rate *(y[i] - z) * X[i, ]
    }

    # compute cost function of the epoch
    cost[n] = sum((y - X %*% w)^2)/2
    cat(" - cost: ", cost[n], " - error: ", error[n], "\n")
  }

  #returning the model
  model = list(epochs = n.iter, cost = log(cost), lrn.rate = lrn.rate,
    error = error, weights = w)

  return(model)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineSGD.predict = function(example, weights) {

  example = c(1, example)
  z = sum(example * obj$weights)
  y = ifelse(z >= 0, +1, -1)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineSGD.regress = function(example, weights) {
  example = c(1, example)
  z = sum(example * obj$weights)
  return(z)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
