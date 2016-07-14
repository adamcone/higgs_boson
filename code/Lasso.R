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
library(ISLR)
library(glmnet)
library(corrplot)

# Lasso Regression: I will fit one lambda and generate one model for each of
# the six data types corresponding to missingness patterns.

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
#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
# M0
#Before I perform elasticnet, I will check for colinearity between the
#independent variables. I'll plot a 30*30 correlation matrix now.
correlation_matrix = cor(M0.training[ ,!names(M0.training) %in% c('EventId',
                                                                  'Weight',
                                                                  'Label')
                                      ]
                         )
# This looks fine to me. I'll proceed with the elasticnet.
corrplot(correlation_matrix, label = FALSE)
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M0.training[, -c(1, ncol(M0.training) - 1)])[, -1]
y = M0.training$Label
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M0.lasso.models = glmnet(x = x,
                      y = y,
                      form = Label ~ x,
                      family = 'binomial',
                      alpha = 1,
                      lambda = seq(from = 10^0,
                                   to = 10^-5,
                                   length.out = 100)
                      )
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M0.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M0 Lasso Regression")
#fitting the ridge regression. Alpha = 0 for lasso regression.
ptm <- proc.time()
M0.ridge.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 0,
                         lambda = seq(from = 10^1,
                                      to = 10^-4,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M0.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M0 Ridge Regression")
#Running 10-fold cross validation.
set.seed(0)
alpha_grid = seq(from = 0,
                 to = 1,
                 length.out = 10)
lambda_grid = seq(from = 10^-5,
                  to = 10^1,
                  length.out = 10)
ptm <- proc.time()
M0.cv.elastic = cv.glmnet(x = x,
                          y = y,
                          type.measure = 'class',
                          family = 'binomial',
                          lambda = lambda_grid,
                          alpha = alpha_grid,
                          nfolds = 10
                          )
proc.time() - ptm
plot(M0.cv.lasso, main = "Lasso Regression\n")
M0.coef = coef(M0.cv.lasso, s = "lambda.min")
plot(2:nrow(M0.coef), abs(as.vector(M0.coef))[-1])
# empirical beta threshold = 0.1. Beta's above this are, to me, significant.
# There are six beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.1
M0.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M0.coef)) {
  if (abs(M0.coef[i]) > beta_threshold) {
    M0.sig.coef.df[counter, ] = c(row.names(M0.coef)[i], M0.coef[i])
    counter = counter + 1
  }
}
M0.sig.coef.df$Beta = as.numeric(M0.sig.coef.df$Beta)
M0.sig.coef.df = arrange(M0.sig.coef.df, desc(abs(Beta)))
#Now, the variables that came up as significant for the full data are, in
#descending order by coefficient magnitude:
#[1] "DER_deltar_tau_lep"     "DER_lep_eta_centrality"
#[3] "DER_pt_ratio_lep_tau"   "PRI_jet_num"           
#[5] "DER_met_phi_centrality" "DER_deltaeta_jet_jet" 
#I'll now plot a logistic regression on these variables only to see how well
#they can, on their own, predict the Labels for which I have full data.

#------------------------------------------------------------------------------
# M1
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M1.training[, -c(1, ncol(M1.training) - 1)])[, -1]
y = M1.training$Label
grid = 10^seq(from = 0,
              to = -5,
              length.out = 1000)
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M1.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = grid
                         )
proc.time() - ptm
#Visualizing the lasso regression shrinkage.
plot(M1.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M1 Lasso Regression")
#Running 10-fold cross validation.
set.seed(0)
ptm <- proc.time()
M1.cv.lasso = cv.glmnet(x = x,
                        y = y,
                        type.measure = 'class',
                        family = 'binomial',
                        lambda = grid,
                        alpha = 1,
                        nfolds = 10
                        )
proc.time() - ptm
plot(M1.cv.lasso, main = "Lasso Regression\n")
M1.coef = coef(M1.cv.lasso, s = "lambda.min")
plot(2:nrow(M1.coef), abs(as.vector(M1.coef))[-1])
# empirical beta threshold = 0.2. Beta's above this are, to me, significant.
# There are four beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.2
M1.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M1.coef)) {
  if (abs(M1.coef[i]) > beta_threshold) {
    M1.sig.coef.df[counter, ] = c(row.names(M1.coef)[i], M1.coef[i])
    counter = counter + 1
  }
}
M1.sig.coef.df$Beta = as.numeric(M1.sig.coef.df$Beta)
M1.sig.coef.df = arrange(M1.sig.coef.df, desc(abs(Beta)))
#------------------------------------------------------------------------------
# M7
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M7.training[, -c(1, ncol(M7.training) - 1)])[, -1]
y = M7.training$Label
grid = 10^seq(from = 0,
              to = -6,
              length.out = 1000)
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M7.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = grid
                         )
proc.time() - ptm
#Visualizing the lasso regression shrinkage.
plot(M7.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M7 Lasso Regression")
#Running 10-fold cross validation.
set.seed(0)
ptm <- proc.time()
M7.cv.lasso = cv.glmnet(x = x,
                        y = y,
                        type.measure = 'class',
                        family = 'binomial',
                        lambda = grid,
                        alpha = 1,
                        nfolds = 10
                        )
proc.time() - ptm
plot(M7.cv.lasso, main = "Lasso Regression\n")
M7.coef = coef(M7.cv.lasso, s = "lambda.min")
plot(2:nrow(M7.coef), abs(as.vector(M7.coef))[-1])
# empirical beta threshold = 0.2. Beta's above this are, to me, significant.
# There are three beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.2
M7.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M7.coef)) {
  if (abs(M7.coef[i]) > beta_threshold) {
    M7.sig.coef.df[counter, ] = c(row.names(M7.coef)[i], M7.coef[i])
    counter = counter + 1
  }
}
M7.sig.coef.df$Beta = as.numeric(M7.sig.coef.df$Beta)
M7.sig.coef.df = arrange(M7.sig.coef.df, desc(abs(Beta)))
#------------------------------------------------------------------------------
# M8
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M8.training[, -c(1, ncol(M8.training) - 1)])[, -1]
y = M8.training$Label
grid = 10^seq(from = 0,
              to = -6,
              length.out = 1000)
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M8.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = grid)
proc.time() - ptm
#Visualizing the lasso regression shrinkage.
plot(M8.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M8 Lasso Regression")
#Running 10-fold cross validation.
set.seed(0)
ptm <- proc.time()
M8.cv.lasso = cv.glmnet(x = x,
                        y = y,
                        type.measure = 'class',
                        family = 'binomial',
                        lambda = grid,
                        alpha = 1,
                        nfolds = 10
)
proc.time() - ptm
plot(M8.cv.lasso, main = "Lasso Regression\n")
M8.coef = coef(M8.cv.lasso, s = "lambda.min")
plot(2:nrow(M8.coef), abs(as.vector(M8.coef))[-1])
# empirical beta threshold = 0.2. Beta's above this are, to me, significant.
# There are three beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.2
M8.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M8.coef)) {
  if (abs(M8.coef[i]) > beta_threshold) {
    M8.sig.coef.df[counter, ] = c(row.names(M8.coef)[i], M8.coef[i])
    counter = counter + 1
  }
}
M8.sig.coef.df$Beta = as.numeric(M8.sig.coef.df$Beta)
M8.sig.coef.df = arrange(M8.sig.coef.df, desc(abs(Beta)))
#------------------------------------------------------------------------------
# M10
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M10.training[, -c(1, ncol(M10.training) - 1)])[, -1]
y = M10.training$Label
grid = 10^seq(from = 1,
              to = -5,
              length.out = 1000)
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M10.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = grid)
proc.time() - ptm
#Visualizing the lasso regression shrinkage.
plot(M10.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M10 Lasso Regression")
#Running 10-fold cross validation.
set.seed(0)
ptm <- proc.time()
M10.cv.lasso = cv.glmnet(x = x,
                         y = y,
                         type.measure = 'class',
                         family = 'binomial',
                         lambda = grid,
                         alpha = 1,
                         nfolds = 10
                         )
proc.time() - ptm
plot(M10.cv.lasso, main = "Lasso Regression\n")
M10.coef = coef(M10.cv.lasso, s = "lambda.min")
plot(2:nrow(M10.coef), abs(as.vector(M10.coef))[-1])
# empirical beta threshold = 1. Beta's above this are, to me, significant.
# There are two beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 1
M10.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M10.coef)) {
  if (abs(M10.coef[i]) > beta_threshold) {
    M10.sig.coef.df[counter, ] = c(row.names(M10.coef)[i], M10.coef[i])
    counter = counter + 1
  }
}
M10.sig.coef.df$Beta = as.numeric(M10.sig.coef.df$Beta)
M10.sig.coef.df = arrange(M10.sig.coef.df, desc(abs(Beta)))
#------------------------------------------------------------------------------
# M11
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M11.training[, -c(1, ncol(M11.training) - 1)])[, -1]
y = M11.training$Label
grid = 10^seq(from = 0,
              to = -6,
              length.out = 1000)
#fitting the lasso regression. Alpha = 1 for lasso regression.
ptm <- proc.time()
M11.lasso.models = glmnet(x = x,
                          y = y,
                          family = 'binomial',
                          alpha = 1,
                          lambda = grid)
proc.time() - ptm
#Visualizing the lasso regression shrinkage.
plot(M11.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M11 Lasso Regression")
#Running 10-fold cross validation.
set.seed(0)
ptm <- proc.time()
M11.cv.lasso = cv.glmnet(x = x,
                         y = y,
                         type.measure = 'class',
                         family = 'binomial',
                         lambda = grid,
                         alpha = 1,
                         nfolds = 10
                         )
proc.time() - ptm
plot(M11.cv.lasso, main = "Lasso Regression\n")
M11.coef = coef(M11.cv.lasso, s = "lambda.min")
plot(2:nrow(M11.coef), abs(as.vector(M11.coef))[-1])
# empirical beta threshold = 0.1. Beta's above this are, to me, significant.
# There are three beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.1
M11.sig.coef.df = data.frame(Variable_Name = 0, Beta = 0)
counter = 1
for (i in 2:nrow(M11.coef)) {
  if (abs(M11.coef[i]) > beta_threshold) {
    M11.sig.coef.df[counter, ] = c(row.names(M11.coef)[i], M11.coef[i])
    counter = counter + 1
  }
}
M11.sig.coef.df$Beta = as.numeric(M11.sig.coef.df$Beta)
M11.sig.coef.df = arrange(M11.sig.coef.df, desc(abs(Beta)))