# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("../perceptron.R")
library("ggplot2")

# -----------------------------------------------------------------
# -----------------------------------------------------------------

seed.value = 42
set.seed(seed.value)

# -----------------------------------------------------------------
# creating dataset
# -----------------------------------------------------------------

# artificial dataset
#x1    = c(4,2,5,3,1.5,2.5,4,5,1.5,3,5,4)
#x2    = c(5,4.5,4.5,4,3,3,3,3,1.5,2,1.5,1)
#bias  = 1
#class = c(1,-1,1,1,-1,-1,1,1,-1,-1,1,-1)
#dataset = data.frame(bias, x1, x2, class)

# dataset AND (uncomment to run with this dataset)
 x1 = c(0,0,1,1)
 x2 = c(0,1,0,1)
 bias  = 1
 class = c(-1,-1,-1,1)
 dataset = data.frame(bias, x1, x2, class)

# -----------------------------------------------------------------
#  training perceptron
# -----------------------------------------------------------------

w = c(0.5, 0.5, 0.5)
obj = perceptron.train(dataset = dataset, weights = w,
  lrn.rate = 0.5)

# -----------------------------------------------------------------
# plotting training error convergence
# -----------------------------------------------------------------

df = data.frame(1:obj$epochs, obj$avgErrorVec)
colnames(df) = c("epoch", "avgError")

# Avg training error
g = ggplot(df, mapping = aes(x = epoch, y = avgError))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
ggsave(g, file = paste0("perceptron_convergence_",
  seed.value,".jpg"), width = 7.95, height = 3.02, dpi = 480)

# -----------------------------------------------------------------
# ploting the obtained hyperplane
# -----------------------------------------------------------------

dataset$class = as.factor(dataset$class)
g2 = ggplot(dataset, mapping = aes(x = x1, y = x2, colour = class,
  shape = class))
g2 = g2 + scale_x_continuous(limit = c(0, 7))
g2 = g2 + scale_y_continuous(limit = c(0, 6))
g2 = g2 + geom_point(size = 3) + theme_bw()

# hyper-plane
slope = -(obj$weights[1]/obj$weights[3])/(obj$weights[1]/obj$weights[2])
intercept = -obj$weights[1]/obj$weights[3]

g2 = g2 + geom_abline(intercept = intercept, slope = slope)
ggsave(g2, file = paste0("perceptron_hyperplane_",
  seed.value,".jpg"), width = 6, height = 6, dpi = 480)

# -----------------------------------------------------------------
# testing data
# -----------------------------------------------------------------

test1 = c(1,2,2)
res1 = perceptron.predict(example = test1, weights = obj$weights)
print(test1)
print(res1)

test2 = c(1,4,4)
res2 = perceptron.predict(example = test2, weights = obj$weights)
print(test2)
print(res2)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
