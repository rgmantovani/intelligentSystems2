# -----------------------------------------------------------------
# -----------------------------------------------------------------

adalineSGD.train = function(dataset, n.iter = 100,
  lrn.rate = 0.0001, classification = TRUE) {

  # we are dealing with extended input and weight vectors
  X = cbind(1, dataset)
  class.id = ncol(X)
  y = as.numeric(X[,class.id])
  X = as.matrix(X[,-class.id])

  # generating random weights [-0.5, 0.5]
  w = matrix(data = runif(n = ncol(dataset), min = -0.5, max = 0.5))

  # initialize vector to keep track of cost and error function per epoch
  error = rep(0, n.iter)
  misc  = rep(0, n.iter)

  # loop through each epoch
  for(n in 1:n.iter) {

    cat("* Epoch:", n)
  
    # loop through each data point
    for(i in sample(x = 1:nrow(dataset), 
      size = nrow(dataset), replace = FALSE)) {

      # keep track of incorrect predictions
      z = sum(w * X[i, ])

      if(classification) {
        ypred = ifelse(z < 0.0, -1, +1)
        # count the number of misclassifications
        if(ypred != y[i]) {
          misc[n] = misc[n] + 1
        }
      } else {
        ypred = z
      }

      # update weights
      w = w + lrn.rate *(y[i] - z) * X[i, ]
    }

    # compute cost function of the epoch
    error[n] = sum((y - X %*% w)^2)/nrow(X)  #2
    cat(" - error: ", error[n], " - misc: ", misc[n], "\n")
  }

  #returning the model
  model = list(epochs = n.iter, misc = misc, lrn.rate = lrn.rate,
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

adalineSGD.regression = function(example, weights) {
  example = c(1, example)
  z = sum(example * obj$weights)
  return(z)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
