# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

source("../evolutivesAux.R")
source("../steadyState.R")
source("../generationBased.R")
source("../evolutionaryProgramming.R")
source("../geneticAlgorithm.R")
set.seed(42)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

mat.dist = make.distances(dmin = 0, dmax = 1000, ncities = 5)
print(mat.dist)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

POPULATION.SIZE = 10000


# test01
obj = steadyState(dist = mat.dist, ncities = nrow(mat.dist),
  population.size = 100, ngenerations = POPULATION.SIZE)

# plot(x = 1:length(obj$avgFitness), y = obj$avgFitness)

best.fitness = max((obj$population[,6]))
ids = which(obj$population[,6] == best.fitness)
obj$population[ids,]

#solucoes unicas
unique(obj$population[ids, ])

# total de solucoes
nrow(unique(obj$population[ids,]))

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

obj2 = generationBased(dist = mat.dist, k.childreen = 10, ncities = nrow(mat.dist),
	population.size = 100, ngenerations = POPULATION.SIZE)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

obj3 = evolutionaryProgramming(dist.matrix = mat.dist, ncities = nrow(mat.dist),
  pop.size = 100, ngenerations = POPULATION.SIZE)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

obj4 = geneticAlgorithm(dist.matrix = mat.dist, selection = "roulette",
  ncities = nrow(mat.dist), pop.size = 100, ngenerations = POPULATION.SIZE,
  mutation.prob = 0.02, elitism = 0.1, k = 2)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# plot(x = 1:length(obj2$avgFitness), y = obj2$avgFitness)

# TODO: plotar as duas curvas (comparar os algoritmo)
df = cbind(1:length(obj$avgFitness), obj$avgFitness, obj2$avgFitness, obj3$avgFitness,
  obj4$avgFitness)
df = data.frame(df)
colnames(df) = c("generation", "SteadyState", "GenerationBased", "EvolProgramming", "GA")
df = reshape2::melt(df, id.vars = 1)
#
library(ggplot2)
g = ggplot(df, mapping = aes(x = generation, y = value, group = variable,
  shape = variable, linetype = variable, colour = variable))
g = g + geom_line()
print(g)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
