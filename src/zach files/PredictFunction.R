setwd('/Users/zacharyescalante/Documents/Kaggle')
load('Kaggle3.RData')
library(e1071)

#M0 Prediction Data Set
M0.predict = sample(1:nrow(M0.training), 10000)
ptm <- proc.time()
M0.svm.full = svm(Label ~.,
             data = M0.training[M0.predict, -c(1, ncol(M0.training)-1)],
             kernel = 'radial',
             cost = cv.svm.M0$best.parameters[1],
             gamma = cv.svm.M0$best.parameters[2],
             probability = TRUE)
proc.time() - ptm

#M1 Prediction Data Set
ptm <- proc.time()
M1.svm.full = svm(Label ~.,
                  data = M1.training[, -c(1, ncol(M1.training)-1)],
                  kernel = 'radial',
                  cost = cv.svm.M1$best.parameters[1],
                  gamma = cv.svm.M1$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M7 Prediction Data Set
M7.predict = sample(1:nrow(M7.training), 10000)
ptm <- proc.time()
M7.svm.full = svm(Label ~.,
                  data = M7.training[M7.predict, -c(1, ncol(M7.training)-1)],
                  kernel = 'radial',
                  cost = cv.svm.M7$best.parameters[1],
                  gamma = cv.svm.M7$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M8 Prediction Data Set
ptm <- proc.time()
M8.svm.full = svm(Label ~.,
                  data = M8.training[, -c(1, ncol(M8.training)-1)],
                  kernel = 'radial',
                  cost = cv.svm.M8$best.parameters[1],
                  gamma = cv.svm.M8$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M10 Prediction Data Set
M10.predict = sample(1:nrow(M10.training), 10000)
ptm <- proc.time()
M10.svm.full = svm(Label ~.,
                  data = M10.training[M7.predict, -c(1, ncol(M10.training)-1)],
                  kernel = 'radial',
                  cost = cv.svm.M10$best.parameters[1],
                  gamma = cv.svm.M10$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm

#M11 Prediction Data Set
M11.predict = sample(1:nrow(M11.training), 10000)
ptm <- proc.time()
M11.svm.full = svm(Label ~.,
                  data = M11.training[M11.predict, -c(1, ncol(M11.training)-1)],
                  kernel = 'radial',
                  cost = cv.svm.M11$best.parameters[1],
                  gamma = cv.svm.M11$best.parameters[2],
                  probability = TRUE)
proc.time() - ptm



predict_function_full = function(test.obs){
  obs.NAs = sum(is.na(test.obs))
  
  if(obs.NAs == 0){
    return(predict(M0.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 1){
    return(predict(M1.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 7){
    return(predict(M7.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 8){
    return(predict(M8.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else if(obs.NAs == 10){
    return(predict(M10.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
  else{
    return(predict(M11.svm.full, test.obs[-1], probability = TRUE, na.rm = TRUE))
  }
}

s.prob = as.data.frame(matrix(ncol = 3, nrow = nrow(test)))
names(s.prob) = c('EventId', 'S.Probability', 'Label')
for (i in 1:nrow(test)){
  x = predict_function_full(test[i,])
  s.prob[i, 1] = test[i, 1]
  s.prob[i, 2] = attr(x, "probabilities")[1,]['s']
  s.prob[i, 3] = as.character(x[[1]][1])
}



