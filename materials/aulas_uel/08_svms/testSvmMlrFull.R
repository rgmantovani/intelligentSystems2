# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

library("mlr")

# iris (Species)
task  = mlr::makeClassifTask(id = "iris", data = iris, target = "Species")
lrn   = mlr::makeLearner(cl = "classif.svm", predict.type = "prob")
rdesc = mlr::makeResampleDesc(method = "CV", iters = 10, stratify = TRUE)
meas  = list(ber, kappa, logloss, multiclass.aunu, timetrain, timepredict, timeboth)

res   =  mlr::resample(learner = lrn, task = task, resampling = rdesc, measures = meas,
  models = FALSE, show.info = TRUE)
# print(res)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# comparing differnt classifiers classifiers
lrn.svm = lrn
lrn.j48 = mlr::makeLearner("classif.J48", predict.type = "prob")
lrn.nb  = mlr::makeLearner("classif.naiveBayes", predict.type = "prob")
lrn.knn = mlr::makeLearner("classif.kknn", predict.type = "prob")
lrn.rf  = mlr::makeLearner("classif.randomForest", predict.type = "prob")
lrn.mlp = mlr::makeLearner("classif.mlp", predict.type = "prob")

rdesc.rep = mlr::makeResampleDesc(method = "RepCV", folds = 10,
  rep = 5, stratify = TRUE)
learners  = list(lrn.svm, lrn.j48, lrn.nb, lrn.knn, lrn.rf, lrn.mlp)

ben1 = mlr::benchmark(learners = learners, tasks = task, resamplings = rdesc.rep,
  measures = meas, models = FALSE, show.info = TRUE)
print(ben1)

mlr::plotBMRBoxplots(bmr = ben1, style = "violin")
mlr::plotBMRSummary(bmr = ben.full)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# several tasks
tasks = list(pid.task)
ben2  = mlr::benchmark(learners = learners, tasks = tasks, resamplings = rdesc.rep,
  measures = meas, models = FALSE, show.info = TRUE)
print(ben2)

ben.full = mlr::mergeBenchmarkResults(bmrs = list(ben1, ben2))

# plots
mlr::plotBMRBoxplots(bmr = ben.full, style = "violin")

mlr::plotBMRSummary(bmr = ben.full)

# statistical tests
stat.data = mlr::generateCritDifferencesData(bmr = ben.full, p.value = 0.05)
mlr::plotCritDifferences(obj = stat.data)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
