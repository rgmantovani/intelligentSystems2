# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("../RBF.R")

library("setwidth")
library("ggplot2")
library("OpenML")
library("mlr")

set.seed(42)

# -----------------------------------------------------------------
# get OpenML data
# -----------------------------------------------------------------

dataset = OpenML::getOMLDataSet(data.id = 61)
data    = dataset$data

X = data[ , c(2,4,5)]
X$class = ifelse(X$class == "Iris-setosa", +1, -1)
X = cbind(1, X)
colnames(X)[1] = "bias"

# view data
plot(X$sepalwidth, X$petalwidth, col=X$class+3)

# -----------------------------------------------------------------
# Training and test data
# -----------------------------------------------------------------

# split into data and test (using mlr)
X$class = as.factor(X$class)
task = makeClassifTask(id = "iris", data = X, target = "class")
res = mlr::makeResampleInstance("CV", task = task, iters = 3,
  stratify = TRUE)

# acessing just first fold (for CV, must iterate all of them !!!)
train.data = X[res$train.inds[[1]], ]
test.data  = X[res$test.inds[[1]], ]

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Running just one CV iteration
# TODO iterate several times

train.data$class = as.numeric(as.character(train.data$class))
test.data$class = as.numeric(as.character(test.data$class))

# train RBF
rbf.model = rbf.train(dataset = train.data, K = 10, gamma = 0.1)

# predict data
rbf.pred = rbf.predict(model = rbf.model,
  X = test.data[, -ncol(test.data)], classification = TRUE)

# checking accuracy
tab = table(test.data$class, rbf.pred)
acc = sum(diag(tab))/nrow(test.data)

print(acc)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# checking results
plot(test.data$sepalwidth, test.data$petalwidth, col=test.data$class+3, pch=0)
points(test.data$sepalwidth, test.data$petalwidth, col=rbf.pred+3, pch=3)

# draw the model centers
points(rbf.model$centers[,-1], col="black", pch=19)
legend("topleft",c("true value","predicted"),pch=c(0,3),bg="white")

# -----------------------------------------------------------------
# -----------------------------------------------------------------
