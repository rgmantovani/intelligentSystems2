# -----------------------------------------------------------------
# -----------------------------------------------------------------

library("ggplot2")
# library(reshape2)

source("../adalineSGD.R")

seed.value = 42
set.seed(seed.value)

# -----------------------------------------------------------------
# getting dataset from OpenML
# -----------------------------------------------------------------

dataset = OpenML::getOMLDataSet(data.id = 61)
data    = dataset$data

dataset = data[ , c(2,4,5)]
dataset$class = ifelse(dataset$class == "Iris-setosa", +1, -1)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

obj = adalineSGD.train(dataset = dataset, n.iter = 1000,
  lrn.rate = 0.001)

obj$weights  = t(obj$weights)
w0 = obj$weights[1]
w1 = obj$weights[2]
w2 = obj$weights[3]

# hyperplane
slope     = -(w0/w2)/(w0/w1)
intercept = -w0/w2

dataset$class = as.factor(dataset$class)
g = ggplot(dataset, mapping = aes(x = sepalwidth, y = petalwidth,
  colour = class, shape = class))
g = g + geom_point(size = 3) + theme_bw()

slope     = -(w0/w2)/(w0/w1)
intercept = -w0/w2

g = g + geom_abline(intercept = intercept, slope = slope)
ggsave(g, file = paste0("adalineSGD_hyperplane_",
  seed.value,".jpg"), width = 6, height = 6, dpi = 480)

# -----------------------------------------------------------------
# convergence
# -----------------------------------------------------------------

df = data.frame(1:obj$epochs, obj$error)

# Ploting convergence
colnames(df) = c("epoch", "Accuracy")
g = ggplot(df, mapping = aes(x = epoch, y = Accuracy))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
ggsave(g, file = paste0("adalineSGD_predErrors_", seed.value,".jpg"),
  width = 7.95, height = 3.02, dpi = 480)

# Plotting the Cost function
df = data.frame(1:obj$epochs, obj$cost)
colnames(df) = c("epoch", "Cost")
g = ggplot(df, mapping = aes(x = epoch, y = Cost))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
ggsave(g, file = paste0("adalineSGD_cost_", seed.value,".jpg"),
  width = 7.95, height = 3.02, dpi = 480)

# -----------------------------------------------------------------
# testing data
# -----------------------------------------------------------------

test1 = c(2.5,0.3)
res1 = adalineSGD.predict(example = test1, weights = obj$weights)
print(test1)
print(res1)

test2 = c(2.5,0.5)
res2 = adalineSGD.predict(example = test2, weights = obj$weights)
print(test2)
print(res2)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
