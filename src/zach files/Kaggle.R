setwd('/Users/zacharyescalante/Documents/Kaggle/')
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
#aggr(training)
#aggr(test)

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

#First SVM Model
library(dplyr)
library(e1071)
#Generate one model for each of the 6 combinations of missingness


#Create 6 separate data frames based on missingness combinations

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
M7.training = select(M7.training, -grep("PRI_jet_num", colnames(M7.training)))
                        #We delete the integer variable which counts the number
                        #of particle since it has a constant value and provides
                        #no predictive value and cannot be scaled.
M8.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
                        filter(., missing_vars == 8) %>%
                        select(., -c(2, 6, 7, 8, 14, 28, 29, 30, 34))
M8.training = select(M8.training, -grep("PRI_jet_num", colnames(M8.training)))
                        #We delete the integer variable which counts the number
                        #of particle since it has a constant value and provides
                        #no predictive value and cannot be scaled.
M10.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
                        filter(., missing_vars == 10) %>%
                        select(., -c(6, 7, 8, 14, 25, 26, 27, 28, 29, 30, 34))
M10.training = select(M10.training, -grep("PRI_jet_num", colnames(M10.training)))
M10.training = select(M10.training, -grep("PRI_jet_all_pt", colnames(M10.training)))
                        #We delete the integer variable which counts the number ("PRI_jet_num")
                        #of particle since it has a constant value and provides
                        #no predictive value and cannot be scaled. We also delete
                        #a measurement based off that variable ("PRI_jet_pt")
M11.training = transform(training,
                        missing_vars = rowSums((is.na(training)))
                        ) %>%
                        filter(., missing_vars == 11) %>%
                        select(., -c(2, 6, 7, 8, 14, 25, 26, 27, 28, 29, 30, 34))
M11.training = select(M11.training, -grep("PRI_jet_num", colnames(M11.training)))
M11.training = select(M11.training, -grep("PRI_jet_all_pt", colnames(M11.training)))
                        #We delete the integer variable which counts the number ("PRI_jet_num")
                        #of particle since it has a constant value and provides
                        #no predictive value and cannot be scaled. We also delete
                        #a measurement based off that variable ("PRI_jet_pt")

#We will use a raidal kernel for all six data frames.
cost = seq(-3, 5, length = 3)
gamma = seq(-5, 1, length = 3)

random.sample.M0 = sample(1:nrow(M0.training), 10000)


set.seed(0)
#0 missing data columns
ptm <- proc.time()
set.seed(0)
M0.index = sample(1:nrow(M0.training), 10000)
cv.svm.M0 = tune(svm,
                Label ~ .,
                data = M0.training[M0.index, -c(1, ncol(M0.training) - 1)],
                kernel = "radial",
                ranges = list(cost = 10^(cost), 
                gamma = 10^(gamma)),          
                tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M0

#1 missing data column
ptm <- proc.time()
cv.svm.M1 = tune(svm,
                 Label ~ .,
                 data = M1.training[, -c(1, ncol(M1.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost),
                               gamma = 10^(gamma)),           
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M1

#7 missing data columns
ptm <- proc.time()
set.seed(0)
M7.index = sample(1:nrow(M7.training), 10000)
cv.svm.M7 = tune(svm,
                 Label ~ .,
                 data = M7.training[M7.index, -c(1, ncol(M7.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost),   #best cost = 0.01. Pass cost = seq(0.01, 2.00, length = 2)
                               gamma = 10^(gamma)),  #best gamma = 0.001. Pass gamma = seq(0.001, 0.1, length = 2)
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M7

#8 missing data columns
ptm <- proc.time()
cv.svm.M8 = tune(svm,
                 Label ~ .,
                 data = M8.training[, -c(1, ncol(M8.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost),        #best cost = 0.01. Pass cost = seq(0.01, 2.00, length = 2)
                               gamma = 10^(gamma)),      #best gamma = 0.001. Pass gamma = seq(0.001, 0.1, length = 2)
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M8

#10 missing data columns
ptm <- proc.time()
set.seed(0)
M10.index = sample(1:nrow(M10.training), 10000)
cv.svm.M10 = tune(svm,
                 Label ~ .,
                 data = M10.training[M10.index, -c(1, ncol(M10.training) - 1)],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost), #best cost = 5.272632. Pass cost = seq(5.272632.*.9, 5.272632.*1.1, length = 2)
                               gamma = 10^(gamma)),              #best gamma = 0.001. Pass gamma = seq(0.001, 0.1, length = 2)
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M10

#11 missing data columns
ptm <- proc.time()
set.seed(0)
M11.index = sample(1:nrow(M11.training), 10000)
cv.svm.M11 = tune(svm,
                 Label ~ .,
                 data = M11.training[, -c(1, 32)],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost),    #best cost = 300. Pass cost = seq(300*.9, 300*1.1, length = 3)
                               gamma = 10^(gamma)),      #best gamma = 0.001. Pass gamma = seq(0.001, 0.1, length = 2)
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.M11

#Variable Reduction Using Elastic Net Regression
M0.Reduced.DF = M0.training[, c('DER_deltar_tau_lep', 'DER_lep_eta_centrality', 'DER_pt_ratio_lep_tau',
                                'PRI_jet_num', 'DER_met_phi_centrality', 'DER_deltaeta_jet_jet', 'Label')]
M1.Reduced.DF = M1.training[, c('DER_lep_eta_centrality', 'PRI_jet_num', 'DER_deltar_tau_lep', 'DER_pt_ratio_lep_tau', 'Label')]
M7.Reduced.DF = M7.training[, c('DER_met_phi_centrality', 'DER_deltar_tau_lep', 'DER_pt_ratio_lep_tau', 'Label')]
M8.Reduced.DF = M8.training[, c('DER_met_phi_centrality', 'DER_deltar_tau_lep', 'DER_pt_ratio_lep_tau', 'Label')]
M10.Reduced.DF = M10.training[, c('DER_deltar_tau_lep', 'DER_pt_ratio_lep_tau', 'Label')]
M11.Reduced.DF = M11.training[, c('DER_met_phi_centrality', 'DER_deltar_tau_lep', 'DER_pt_ratio_lep_tau', 'Label')]

#Tune SVM parameters for reduced variable training sets
#M0 Reduced Dataframe
ptm <- proc.time()
cv.svm.reduced.M0 = tune(svm,
                 Label ~ .,
                 data = M0.Reduced.DF[M0.index,],
                 kernel = "radial",
                 ranges = list(cost = 10^(cost), 
                               gamma = 10^(gamma)),          
                 tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M0

#M1 Reduced Datafram
ptm <- proc.time()
cv.svm.reduced.M1 = tune(svm,
                         Label ~ .,
                         data = M1.Reduced.DF[,],
                         kernel = "radial",
                         ranges = list(cost = 10^(cost), #best cost = 10.535263. Pass cost = seq(10.5*0.9, 10.5*1.1, length = 3)
                                       gamma = 10^(gamma)),              #best gamma = 0.01. Pass gamma = seq(0.05, 0.015, length = 5)
                         tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M1

#M7 Reduced Dataframe
ptm <- proc.time()
cv.svm.reduced.M7 = tune(svm,
                         Label ~ .,
                         data = M7.Reduced.DF[M7.index,],
                         kernel = "radial",
                         ranges = list(cost = 10^(cost), 
                                       gamma = 10^(gamma)),              
                         tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M7

#M8 Reduced Dataframe
ptm <- proc.time()
cv.svm.reduced.M8 = tune(svm,
                         Label ~ .,
                         data = M8.Reduced.DF[,],
                         kernel = "radial",
                         ranges = list(cost = 10^(cost), 
                                       gamma = 10^(gamma)),              
                         tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M8


#M10 Reduced Dataframe
ptm <- proc.time()
cv.svm.reduced.M10 = tune(svm,
                         Label ~ .,
                         data = M10.Reduced.DF[M10.index,],
                         kernel = "radial",
                         ranges = list(cost = 10^(cost), 
                                       gamma = 10^(gamma)),              
                         tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M10

#M11 Reduced Dataframe
ptm <- proc.time()
cv.svm.reduced.M11 = tune(svm,
                          Label ~ .,
                          data = M11.Reduced.DF[M11.index,],
                          kernel = "radial",
                          ranges = list(cost = 10^(cost), 
                                        gamma = 10^(gamma)),           
                          tunecontrol = tune.control(sampling = 'cross', cross = 2))
proc.time() - ptm
cv.svm.reduced.M11
