# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("mlp.R")

library("setwidth")
library("ggplot2")
library("OpenML")
library("mlr")

set.seed(42)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Balance sclase dataset
# https://www.openml.org/d/11

dataset = OpenML::getOMLDataSet(data.id = 11)
data    = dataset$data
task    = makeClassifTask(id = "bal_scale", data = data,
  target = "class")

# -----------------------------------------------------------------
# -----------------------------------------------------------------

#split class into 3 target variables
new.data = data
new.data$class.B = 0
new.data$class.R = 0
new.data$class.L = 0

new.data$class.B[which(new.data$class == "B")] = 1
new.data$class.R[which(new.data$class == "R")] = 1
new.data$class.L[which(new.data$class == "L")] = 1

new.data$class = NULL

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# split into data and test (using mlr)
task    = makeClassifTask(id = "bal_scale", data = data,
  target = "class")
res = mlr::makeResampleInstance("CV", task = task, stratify = TRUE)
train.data = new.data[res$train.inds[[1]], ]
test.data  = new.data[res$test.inds[[1]], ]

# -----------------------------------------------------------------
# -----------------------------------------------------------------

#  activation function
fnet = function(x) {
  y = 1 /(1 + exp(-x))
  return(y)
}

# derivative function
dfnet = function(x) {
  y = x * (1-x)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

model = mlp.create(input.length = 4, hidden.length = 5,
  output.length = 3, fnet = fnet, dfnet = dfnet)

obj = mlp.train(model = model, dataset = train.data, lrn.rate = 0.1,
    threshold = 1e-2, n.iter = 10000)

# test each example from testing set
aux = lapply(1:nrow(test.data), function(i) {
  pred = mlp.test(model = obj$model, example = test.data[i, 1:model$input.length])
  ret = as.numeric(round(pred$fnet.output))
  return(ret)
})

# real class values
y.real = test.data[, 5:7]

# getting predictions
y.pred = data.frame(do.call("rbind", aux))
colnames(y.pred) = colnames(y.real)

# checking predictions x expected classes
acc = 0
for(i in 1:nrow(test.data)){
  if( which.max(y.pred[i,]) == which.max(y.real[i,])) {
    acc = acc + 1
  }
}

res = acc/nrow(test.data)
print(res)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

df = data.frame(1:length(obj$errorVec), obj$errorVec)
colnames(df) = c("epochs", "error")
g = ggplot(df, aes(x = epochs, y = error)) + geom_line()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
