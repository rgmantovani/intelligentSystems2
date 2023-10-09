# -----------------------------------------------------------------
# -----------------------------------------------------------------

# activation function
act.function = function(x, mu, gamma) {

  # F is the eucliadian norm of the vector (norm)
  distance = norm(as.matrix(x - mu),"F")^2
  phi = exp(-gamma * distance)
  return(phi)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# returns a rbf model given the:
# * observations x1, xN of dataset D
# * output value for each observation
# * number of centers
# * gamma value

rbf.train = function(dataset, K = 10, gamma = 1.0) {

  X = dataset[, -ncol(dataset)]
  Y = dataset[,  ncol(dataset)]

  N     = dim(X)[1] # number of observations
  ncols = dim(X)[2] # number of variables

  # let's cluster K centers out of the dataset, and
  # only accept if there are no empty clusters
  repeat {
    km = kmeans(x = X, centers = K)
    if (min(km$size)>0)
      break
  }
  #
  # acess the clusters points
  mus = km$centers

  # initialize Phi matrix
  Phi = matrix(rep(NA,(K+1)*N), ncol=K+1)

  # compute activation function of all combinations of X, and phi
  for (i in 1:N) {
    Phi[i,1] = 1    # bias column
    for (j in 1:K) {
      Phi[i, j+1] = act.function(x = X[i,], mu = mus[j,], gamma = 1)
    }
  }

  #  find RBF weights by interpolation
  # (t(Phi) * Phi)^-1 * t(Phi) * Y
  w = corpcor::pseudoinverse(t(Phi) %*% Phi) %*% t(Phi) %*% Y

  # return model
  model = list(weights = w, centers = mus, gamma = gamma)
  return(model)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

rbf.predict = function(model, X, classification = FALSE) {

  # local variables copying model values
  gamma   = model$gamma
  centers = model$centers
  w       = model$weights
  N       = dim(X)[1]

  # we need to init to a value, so let's start with the bias
  pred = rep(w[1],N)

  for (j in 1:N) {
    # find prediction for point xj
    for (k in 1:length(centers[,1])) {

      # the weight for center[k] is given by w[k+1] (because w[1] is the bias)
      phi = act.function(x = X[j,], mu = centers[k,], gamma = 1)
      pred[j]  = pred[j] + w[k+1] * phi
    }
  }

  if (classification) {
    pred = unlist(lapply(pred, sign))
  }

  return(pred)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
