# -----------------------------------------------------------------------------
# Installing R packages
# keras: loading mnist dataset
# -----------------------------------------------------------------------------
packages = c("keras", "tensorflow")
for(pkg in packages) {
  if(!(pkg %in% rownames(installed.packages()))) {
    cat("Installing: ", pkg, "\n")
    install.packages(pkg)
  }
}

# tensoflow::install_tensorflow()

# -----------------------------------------------------------------------------
# Loading Mnist dataset
# -----------------------------------------------------------------------------
mnist = keras::dataset_mnist()

# -----------------------------------------------------------------------------
# training info
# -----------------------------------------------------------------------------
trainImages = mnist$train$x
trainLabels = mnist$train$y

# -----------------------------------------------------------------------------
# testing info
# -----------------------------------------------------------------------------
testImages = mnist$test$x
testLabels = mnist$test$y

# -----------------------------------------------------------------------------
# Training set dimensions
# -----------------------------------------------------------------------------
print(dim(trainImages))
print(dim(trainLabels))

# -----------------------------------------------------------------------------
# Testing set dimensions
# -----------------------------------------------------------------------------
print(dim(testImages))
print(dim(testLabels))

# -----------------------------------------------------------------------------
# first training example, raw data
# -----------------------------------------------------------------------------
tail(trainImages[1,,])

# ------------------------------------------------------------------------------
# plotting first example as an image
# -----------------------------------------------------------------------------
plot(grDevices::as.raster(trainImages[2,,],max=255))
print(trainLabels[2])

table(trainLabels)
table(testLabels)

# -----------------------------------------------------------------------------
# Sigmoidal activation function
# -----------------------------------------------------------------------------
activationFunction = function(v){
  y = 1 / (1 + exp(-v))
  return(y)
}

# -----------------------------------------------------------------------------
# Derivada da função de ativação
# -----------------------------------------------------------------------------
derivativeActivationFunction = function(y){
  dy = y * (1 -y)
  return(dy)
}

# -----------------------------------------------------------------------------
# Instantiate an Autoencoder model (not trained yet)
# -----------------------------------------------------------------------------
createAEModel = function(inputSize, hiddenSize, actFunction, derivativeFunction)
{

  model = list()

  # Model's parameters
  model$inputSize          = inputSize
  model$hiddenSize         = hiddenSize
  model$actFunction        = actFunction
  model$derivativeFunction = derivativeFunction

  # Hidden Weights: input -> hidden layer
  n.hiddenWeights     = (inputSize + 1) * hiddenSize
  model$hiddenWeights = matrix(runif(min = -0.5,max = 0.5, n.hiddenWeights),
                          nrow = hiddenSize, ncol = inputSize+1)

  # Output Weights: hidden layer -> output
  n.outputWeights = (hiddenSize + 1) * inputSize
  model$outputWeights = matrix(runif(min=-0.5, max=0.5, n.outputWeights),
                      nrow = inputSize, ncol = hiddenSize + 1)

  return(model)
}

# -----------------------------------------------------------------------------
# Feed forwared step in the AE
# -----------------------------------------------------------------------------
AEForward = function(model, example) {

  # input -> hidden
  vHiddenLayer = model$hiddenWeights %*% as.numeric(c(example,1))
  yHiddenLayer = model$actFunction(vHiddenLayer)

  # hidden -> output
  vOutputLayer = model$outputWeights %*% c(yHiddenLayer,1)
  yOutputLayer = model$actFunction(vOutputLayer)

  # Results (signals)
  results = list()
  results$vHiddenLayer = vHiddenLayer
  results$yHiddenLayer = yHiddenLayer
  results$vOutputLayer = vOutputLayer
  results$yOutputLayer = yOutputLayer

  return(results)
}

# -----------------------------------------------------------------------------
# Training AEs
# -----------------------------------------------------------------------------

AETraining = function(model, trainImages, learning.rate = 0.01,
  error.threshold = 0.01, n.epochs = 1000){

  loss = 10
  previous.loss = 0
  epochs = 0

  # Training while loss difference is higher than an error.threshold
  while( ((abs(loss - previous.loss))  >  error.threshold) && (epochs < n.epochs)){

    previous.loss = loss
    loss = 0

    # For all the images in the trainingSet
    for(i in 1:nrow(trainImages)){

      x = trainImages[i,,]/255
      # binarize the image
      x = as.vector(ifelse(x >= 0.5, 1, 0))
      
      # Calculates the output singals
      results = AEForward(model = model, example = x)
      y = results$yOutputLayer

      # Error
      error = x - y

      # loss: binary cross entropy (cumulated for all images)
      loss  = loss + -(sum( x * log(y) + (1-x) * log(1-y)))
      # loss: mean squared error
      # loss = loss + sum((x - y)^2)/2
     
      # Output gradients (deltaO)
      outputGradients = error * model$derivativeFunction(y)

      # Hidden gradients (deltaW)
      outputWs = model$outputWeights[, 1:model$hiddenSize]
      hiddenGradients = as.numeric(model$derivativeFunction(results$yHiddenLayer)) *
          (as.vector(outputGradients) %*% outputWs)

      # Updating weights
      model$outputWeights = model$outputWeights + learning.rate *
        (outputGradients %*% c(results$yHiddenLayer,1))
      model$hiddenWeights = model$hiddenWeights + learning.rate *
        (t(hiddenGradients) %*% as.numeric(c(x,1)))
    }

    loss = loss/nrow(trainImages)
    cat("* Epoch: ", epochs,"/ Loss =",loss,"/",abs(loss-previous.loss),"\n")
    epochs = epochs + 1
  }

  obj = list()
  obj$model = model
  obj$epochs = epochs

  return(obj)
}

# -----------------------------------------------------------------------------
# Predicting and plotting an example
# -----------------------------------------------------------------------------

AEPredict = function(model, example) {
  preds = AEForward(model = model, example = example)
  ret = preds$yOutputLayer 
  ret = as.vector(ifelse(ret >= 0.5, 1, 0))
  # plot(grDevices::as.raster(matrix(ret, ncol = 28)))
  return(ret)
}

# -----------------------------------------------------------------------------
# Testing Autoencoder
# -----------------------------------------------------------------------------

inputDimension = length(trainImages[1,,])
cat("Original data size = ", inputDimension, "features \n")

abstractFeatures = round(inputDimension * 0.1)
cat("Abstract features  = ", abstractFeatures, "features \n")

# Creating an AE model
aeModel = createAEModel(
   inputSize = inputDimension, 
   hiddenSize = abstractFeatures ,
   actFunction = activationFunction,
   derivativeFunction = derivativeActivationFunction
)

# training AEs
obj = AETraining(model = aeModel, trainImages = trainImages[1:10000, , ], 
  learning.rate = 0.5, error.threshold = 0.1, n.epochs = 1000)

# -----------------------------------------------------------------------------
# Plotting AEs latent features
# -----------------------------------------------------------------------------

library("ggplot2")

total.images = 5000
aux = lapply(1:total.images, function(k) {
  pred = AEForward(model = obj$model, example = testImages[k,,])
  return(t(pred$vHiddenLayer))
})

df = data.frame(do.call("rbind", aux))
df = cbind(df[,1:2], testLabels[1:total.images])
colnames(df) = c("AE1", "AE2", "Class")
df$Class = as.factor(df$Class)
ggplot(df, aes(x = AE1, y = AE2, colour = Class)) + geom_point()

# -----------------------------------------------------------------------------
# Plotting images
# -----------------------------------------------------------------------------

plottingImages = function(model, img) {
    image = img/255
    plot(c(0, 200), c(0, 100), type = "n", xlab = "", ylab = "")
    rasterImage(as.raster(image), 0, 0, 100, 100, interpolate=F)
    ret = AEPredict(model = model, example = image) 
    rasterImage(as.raster(matrix(ret, ncol = 28)), 100, 0, 200, 100, interpolate=F)
}

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
