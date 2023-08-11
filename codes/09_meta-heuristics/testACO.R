# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

library("acotsp")
library("netgen")

x = generateRandomNetwork(n.points = 10L)
print(x)
autoplot(x)

ctrl = makeACOTSPControl(alpha = 1.2, beta = 1.8, n.ants = 20L,
  max.iter = 50L, trace.all = TRUE)
res = runACOTSP(x, ctrl, monitor = makeConsoleMonitor())
acotsp:::plotResult(object = res)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
