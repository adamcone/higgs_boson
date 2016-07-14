#Prediction on the Reduced Data Set

setwd('/Users/zacharyescalante/Documents/Kaggle')
load('Kaggle3.RData')
library(e1071)
library(dplyr)

#M0 Prediction Data Set
set.seed(0)
M0.predict = sample(1:nrow(M0.Reduced.DF), 10000)
ptm <- proc.time()
M0.svm.reduced = svm(Label ~.,
                  data = M0.Reduced.DF[M0.predict,],
                  kernel = 'radial',
                  cost = cv.svm.reduced.M0$best.parameters[1],
                  gamma = cv.svm.reduced.M0$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M1 Prediction Data Set
ptm <- proc.time()
M1.svm.reduced = svm(Label ~.,
                  data = M1.Reduced.DF[,],
                  kernel = 'radial',
                  cost = cv.svm.reduced.M1$best.parameters[1],
                  gamma = cv.svm.reduced.M1$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M7 Prediction Data Set
M7.predict = sample(1:nrow(M7.Reduced.DF), 10000)
ptm <- proc.time()
M7.svm.reduced = svm(Label ~.,
                  data = M7.Reduced.DF[M7.predict,],
                  kernel = 'radial',
                  cost = cv.svm.reduced.M7$best.parameters[1],
                  gamma = cv.svm.reduced.M7$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M8 Prediction Data Set
ptm <- proc.time()
M8.svm.reduced = svm(Label ~.,
                  data = M8.Reduced.DF[,],
                  kernel = 'radial',
                  cost = cv.svm.reduced.M8$best.parameters[1],
                  gamma = cv.svm.reduced.M8$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M10 Prediction Data Set
M10.predict = sample(1:nrow(M10.Reduced.DF), 10000)
ptm <- proc.time()
M10.svm.reduced = svm(Label ~.,
                   data = M8.Reduced.DF[M10.predict, ],
                   kernel = 'radial',
                   cost = cv.svm.reduced.M10$best.parameters[1],
                   gamma = cv.svm.reduced.M10$best.parameters[2],
                   probability = TRUE)
proc.time() - ptm

#M11 Prediction Data Set
M11.predict = sample(1:nrow(M11.Reduced.DF), 10000)
ptm <- proc.time()
M11.svm.reduced = svm(Label ~.,
                   data = M11.Reduced.DF[M11.predict, ],
                   kernel = 'radial',
                   cost = cv.svm.reduced.M11$best.parameters[1],
                   gamma = cv.svm.reduced.M11$best.parameters[2],
                   probability = TRUE)
proc.time() - ptm



predict_function_reduced = function(test.obs){
  obs.NAs = sum(is.na(test.obs))
  
  if(obs.NAs == 0){
    return(predict(M0.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 1){
    return(predict(M1.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 7){
    return(predict(M7.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 8){
    return(predict(M8.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 10){
    return(predict(M10.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else{
    return(predict(M11.svm.reduced, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
}

s.prob.reduce = as.data.frame(matrix(ncol = 3, nrow = nrow(test)))
names(s.prob.reduce) = c('EventId', 'S.Probability', 'Label')
for (i in 1:nrow(test)){
  x = predict_function_reduced(test[i,])
  s.prob.reduce[i, 1] = test[i, 1]
  s.prob.reduce[i, 2] = attr(x, "probabilities")[1,]['s']
  s.prob.reduce[i, 3] = as.character(x[[1]][1])
}



