setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
training = read.table('training.csv',
                      sep = ',',
                      header = TRUE,
                      na.strings = c('-999.0')
                      )
test = read.table('test.csv',
                  sep = ',',
                  header = TRUE,
                  na.strings = c('-999.0')
                  )

library(VIM)
library(dplyr)
library(e1071)
aggr(training)
aggr(test)

b_index = training$Label == 'b'
s_index = training$Label == 's'
n_b = sum(b_index)
n_s = sum(s_index)
n_s/(n_b + n_s)
N_b = sum(training$Weight[b_index])
N_s = sum(training$Weight[s_index])

AMS = function(s, b) {
  return(sqrt(2*((s + b + 10)*log(1 + s/(b + 10)) - s)))
}

s = 500
b = 3000
AMS(s, b)

# SVM: we will generate one model for each of the six combinations of
# missingness

M0.training = transform(training,
                          missing_vars = rowSums((is.na(training)))
                          ) %>%
                filter(., missing_vars == 0) %>%
                select(., -34)

M1.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
              filter(., missing_vars == 1) %>%
              select(., -c(2, 34))

M7.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
              filter(., missing_vars == 7) %>%
              select(., -c(6, 7, 8, 14, 28, 29, 30, 34))
M7.training = select(M7.training, -grep('PRI_jet_num', colnames(M7.training)))
              #removed PRI_jet_num because it is constant at 1
              # for this missingness category. Therefore it both
              # provides no predictive value and cannot be scaled.

M8.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
              filter(., missing_vars == 8) %>%
              select(., -c(2, 6, 7, 8, 14, 28, 29, 30, 34))
M8.training = select(M8.training, -grep('PRI_jet_num', colnames(M8.training)))
#removed PRI_jet_num because it is constant at 1
# for this missingness category. Therefore it both
# provides no predictive value and cannot be scaled.

M10.training = transform(training,
                         missing_vars = rowSums((is.na(training)))
                         ) %>%
               filter(., missing_vars == 10) %>%
               select(., -c(6, 7, 8, 14, 25, 26, 27, 28, 29, 30, 34))
M10.training = select(M10.training, -grep('PRI_jet_num', colnames(M10.training)))
M10.training = select(M10.training, -grep('PRI_jet_all_pt', colnames(M10.training)))
#removed PRI_jet_num and PRI_jet_all_pt because both are constant at 0
# for this missingness category. Therefore neither
# provides no predictive value and cannot be scaled.


M11.training = transform(training,
                         missing_vars = rowSums((is.na(training)))
                         ) %>%
               filter(., missing_vars == 11) %>%
               select(., -c(2, 6, 7, 8, 14, 25, 26, 27, 28, 29, 30, 34))
M11.training = select(M11.training, -grep('PRI_jet_num', colnames(M11.training)))
M11.training = select(M11.training, -grep('PRI_jet_all_pt', colnames(M11.training)))
#removed PRI_jet_num and PRI_jet_all_pt because both are constant at 0
# for this missingness category. Therefore neither
# provides no predictive value and cannot be scaled.

# all six svms will use radial kernels. So, each (C, gamma) combination will have
# to be tuned.

# M0
set.seed(0)
ptm <- proc.time()
cv.svm.M0 = tune(method = svm,
                 train.x = Label ~ .,
                 data = M0.training[1:500, -c(1, ncol(M0.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(seq(-1, 1.5,
                                              length = 1)),
                               gamma = 10^(seq(-2, 1,
                                               length = 1)
                                           )
                               ),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 2
                 )
)
proc.time() - ptm
# M1
set.seed(0)
ptm <- proc.time()
cv.svm.M1 = tune(method = svm,
                 train.x = Label ~ .,
                 data = M1.training[1:500, -c(1, ncol(M1.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(seq(-1, 1.5,
                                              length = 1)),
                               gamma = 10^(seq(-2, 1,
                                               length = 1)
                               )
                 ),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 2
                 )
)
proc.time() - ptm
# M7
set.seed(0)
ptm <- proc.time()
cv.svm.M7 = tune(method = svm,
                 train.x = Label ~ .,
                 data = M7.training[1:500, -c(1, ncol(M7.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(seq(-1, 1.5,
                                              length = 1)),
                               gamma = 10^(seq(-2, 1,
                                               length = 1)
                               )
                 ),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 2
                 )
)
proc.time() - ptm
# M8
set.seed(0)
ptm <- proc.time()
cv.svm.M8 = tune(method = svm,
                 train.x = Label ~ .,
                 data = M8.training[1:500, -c(1, ncol(M8.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(seq(-1, 1.5,
                                              length = 1)),
                               gamma = 10^(seq(-2, 1,
                                               length = 1)
                               )
                 ),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 2
                 )
)
proc.time() - ptm
# M10
set.seed(0)
ptm <- proc.time()
cv.svm.M10 = tune(method = svm,
                 train.x = Label ~ .,
                 data = M10.training[1:500, -c(1, ncol(M10.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(seq(-1, 1.5,
                                              length = 1)),
                               gamma = 10^(seq(-2, 1,
                                               length = 1)
                               )
                 ),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 2
                 )
)
proc.time() - ptm
set.seed(0)
# M11
set.seed(0)
ptm <- proc.time()
cv.svm.M11 = tune(method = svm,
                  train.x = Label ~ .,
                  data = M11.training[1:500, -c(1, ncol(M11.training) - 1)],
                  kernel = "radial",
                  ranges = list(cost = 10^(seq(-1, 1.5,
                                               length = 1)),
                                gamma = 10^(seq(-2, 1,
                                                length = 1)
                                )
                  ),
                  tunecontrol = tune.control(sampling = 'cross',
                                             cross = 2
                  )
)
proc.time() - ptm