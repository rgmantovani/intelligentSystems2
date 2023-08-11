# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("../adalineGD.R")
source("../adalineSGD.R")

library("ggplot2")
library("reshape2")


seed.value = 1
set.seed(seed.value)

# -----------------------------------------------------------------
# getting dataset from OpenML
# -----------------------------------------------------------------

dataset = OpenML::getOMLDataSet(data.id = 61)
data    = dataset$data

dataset = data[ , c(2,4,5)]
dataset$class = ifelse(dataset$class == "Iris-setosa", +1, -1)

# -----------------------------------------------------------------
# training adalineGD
# -----------------------------------------------------------------

model.gd  = adalineGD.train(dataset  = dataset, lrn.rate = 0.001,  n.iter = 1000)
model.sgd = adalineSGD.train(dataset = dataset, lrn.rate = 0.001,  n.iter = 1000)

# -----------------------------------------------------------------
# comparing convergence  between GD x SGD
# -----------------------------------------------------------------

df = data.frame(1:model.gd$epochs, model.gd$cost, model.sgd$cost)
colnames(df) = c("epoch", "GD.001", "SGD.001")

df = melt(df, id.vars = 1)
g = ggplot(df, mapping = aes(x = epoch, y = (value), group = variable,
  colour = variable, shape = variable))
g = g + geom_line() + geom_point()
ggsave(g, file = "adaline_cost_comparisons.jpg", width = 7.1,
  height = 3.5)

g = g + scale_x_continuous(limit = c(1, 25))
ggsave(g, file = "adaline_cost_comparisons_scaled.jpg", width = 7.1,
  height = 3.5)

# -----------------------------------------------------------------
# comparing different learning rates for GD
# -----------------------------------------------------------------

model.gd1m = adalineGD.train(dataset  = dataset, lrn.rate = 0.0001, n.iter = 1000)
model.gd1c = adalineGD.train(dataset  = dataset, lrn.rate = 0.0005, n.iter = 1000)
model.gd1d = adalineGD.train(dataset  = dataset, lrn.rate = 0.001,  n.iter = 1000)

df = data.frame(1:model.gd1m$epochs, model.gd1m$cost, model.gd1c$cost,
  model.gd1d$cost)
colnames(df) = c("epoch", "GD.0001", "GD.0005", "GD.001")

df = melt(df, id.vars = 1)
g = ggplot(df, mapping = aes(x = epoch, y = (value), group = variable,
  colour = variable, shape = variable))
g = g + geom_line() + geom_point()
ggsave(g, file = "adaline_gd_convergence_comparisons.jpg", width = 7.1,
  height = 3.5)

g = g + scale_x_continuous(limit = c(1, 200))
ggsave(g, file = "adaline_gd_convergence_comparisons_scaled.jpg", width = 7.1,
  height = 3.5)


# -----------------------------------------------------------------
# comparing different learning rates for SGD
# -----------------------------------------------------------------

model.gd1m = adalineSGD.train(dataset  = dataset, lrn.rate = 0.0001, n.iter = 1000)
model.gd1c = adalineSGD.train(dataset  = dataset, lrn.rate = 0.0005, n.iter = 1000)
model.gd1d = adalineSGD.train(dataset  = dataset, lrn.rate = 0.001,  n.iter = 1000)

df = data.frame(1:model.gd1m$epochs, model.gd1m$cost, model.gd1c$cost,
  model.gd1d$cost)
colnames(df) = c("epoch", "SGD.0001", "SGD.0005", "SGD.001")

df = melt(df, id.vars = 1)
g = ggplot(df, mapping = aes(x = epoch, y = (value), group = variable,
  colour = variable, shape = variable))
g = g + geom_line() + geom_point()
ggsave(g, file = "adaline_sgd_convergence_comparisons.jpg", width = 7.1,
  height = 3.5)

g = g + scale_x_continuous(limit = c(1, 200))
ggsave(g, file = "adaline_sgd_convergence_comparisons_scaled.jpg", width = 7.1,
  height = 3.5)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
