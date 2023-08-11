# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

library("mlr")

# iris (Species)
task  = mlr::makeClassifTask(id = "iris", data = iris,
  target = "Species")
# ou
# task = iris.task

lrn   = mlr::makeLearner(cl = "classif.svm",
  predict.type = "prob") #gaussian kernel

rdesc = mlr::makeResampleDesc(method = "CV",
  iters = 10, stratify = TRUE)

meas  = list(ber, kappa, logloss, multiclass.aunu,
  timetrain, timepredict, timeboth)

res   =  mlr::resample(learner = lrn, task = task,
  resampling = rdesc, measures = meas,
  models = FALSE, show.info = TRUE)
print(res)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
