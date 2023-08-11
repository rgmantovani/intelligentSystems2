# -----------------------------------------------------------------
# -----------------------------------------------------------------

library(ggplot2)
library(reshape2)

seed.value = 1
set.seed(seed.value)

source("../adalineGD.R")

# -----------------------------------------------------------------
# -----------------------------------------------------------------

x1 = c(-1,-1,1,1)
x2 = c(-1,1,-1,1)
class  = c(-1,-1,-1,1)
dataset = data.frame(x1, x2, class)
# dataset$class = as.factor(dataset$class)

# -----------------------------------------------------------------
# training adalineGD
# -----------------------------------------------------------------

obj = adalineGD.train(dataset = dataset, lrn.rate = 0.001,
  classification = TRUE)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

obj$weights = t(obj$weights)

w0 = obj$weights[1]
w1 = obj$weights[2]
w2 = obj$weights[3]

slope     = -(w0/w2)/(w0/w1)
intercept = -w0/w2

g = ggplot(dataset, mapping = aes(x = x1, y = x2,
  colour = as.factor(class), shape = as.factor(class)))
g = g + geom_point(size = 3) + theme_bw()
g = g + geom_abline(intercept = intercept, slope = slope)
ggsave(g, file = paste0("adalineGD_AND_hyperplane_",
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
ggsave(g, file = paste0("adalineGD_AND_predErrors_", seed.value,".jpg"),
  width = 7.95, height = 3.02, dpi = 480)

# Plotting the Cost function
df = data.frame(1:obj$epochs, obj$cost)
colnames(df) = c("epoch", "Cost")
g = ggplot(df, mapping = aes(x = epoch, y = Cost))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
ggsave(g, file = paste0("adalineGD_AND_cost_", seed.value,".jpg"),
  width = 7.95, height = 3.02, dpi = 480)

# -----------------------------------------------------------------
# testing data
# -----------------------------------------------------------------

test1 = c(0,0)
res1 = adalineGD.predict(example = test1, weights = obj$weights)
print(test1)
print(res1)

test2 = c(0,1)
res2 = adalineGD.predict(example = test2, weights = obj$weights)
print(test2)
print(res2)

test3 = c(1,0)
res3 = adalineGD.predict(example = test3, weights = obj$weights)
print(test3)
print(res3)

test4 = c(1,1)
res4 = adalineGD.predict(example = test4, weights = obj$weights)
print(test4)
print(res4)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
