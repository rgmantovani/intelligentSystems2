# -----------------------------------------------------------------
# -----------------------------------------------------------------

perceptron.train = function(dataset, weights = NULL, lrn.rate = 0.3,
  n.iter = 1000) {

  epochs = 0
  error  = TRUE

  if(is.null(weights)) {
    weights = runif(ncol(dataset)-1,-1,1)
  }

  cat("Initial weights: ", weights, "\n")
  avgErrorVec = c()
  class.id = ncol(dataset)

  # while there is an error in training examples
  while(error) {

    # limiting the number of epochs
    if(epochs > n.iter) {
      break
    }


    error  = FALSE
    epochs = epochs + 1
    avgError = 0
    cat("Epoca:", epochs,"\n")

    for(i in 1:nrow(dataset)) {

      example  = as.numeric(dataset[i,])

      # spike
      x = example[-class.id]
      v = as.numeric(x %*% weights)

      # output
      # y = sign(v)
      # it could also be:
      y = ifelse(v >=0, +1, -1)

      avgError = avgError + ((example[class.id] - y)^2)

      # updating weights (only to misclassified patterns)
      if(example[class.id] != y) {
        error = TRUE
        weights = weights + lrn.rate * (example[class.id] - y) * example[-class.id]
      }
      print(weights)
    }

    avgError = avgError/nrow(dataset)
    avgErrorVec = c(avgErrorVec, avgError)
    cat("Epoch: ", epochs," - Avg Error = ", avgError, "\n")
  }

  # returning object with some slots
  obj = list(weights = weights, avgErrorVec = avgErrorVec, epochs = epochs)

  cat("\n* Finished after: ",epochs,"  epochs\n")
  return(obj)
}

# -----------------------------------------------------------------
# Evaluating new examples after the model was trained
# -----------------------------------------------------------------

perceptron.predict = function(example, weights) {

  v = as.numeric(example %*% weights)
  # it also works as:
  # v = sum(example * weights)
  # y = sign(v)
  y = ifelse(v >=0, +1, -1)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
