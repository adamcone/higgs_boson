setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M0: no missing data in these observations: 30 independent variables

#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
elasticnet.trainControl = trainControl(method = 'cv',
                                       number = 10
                                       )

#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M0.training[, -c(1, ncol(M0.training) - 1)])[, -1]
y = M0.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M0.lasso.models = glmnet(x = x,
                         y = y,
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
# performing ridge regression to further hone lambda range.
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
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M0.best.model <- train(x = M0.training[, !names(M0.training) %in% c('EventId', 'Weight', 'Label')],
                       y = M0.training[, 'Label'],
                       method='glmnet',
                       metric = "Accuracy",
                       preProc = c('center', 'scale'),
                       tuneGrid = expand.grid(.alpha=seq(from = 0,
                                                         to = 1,
                                                         length.out = 10),
                                              .lambda = 10^seq(from = -5,
                                                               to = 1,
                                                               length.out = 10)
                       ),
                       trControl = elasticnet.trainControl
                       )
proc.time() - ptm
# This took ten minutes to run. With 100 (alpha, lambda) combinations, the best
# tuning parameters were (alpha = 0.1111111, lambda = 0.001), which resulted in
# a cross-validation accuracy of 0.7288516.
M0.alpha = M0.best.model$bestTune$alpha
M0.lambda = M0.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M0.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M0.alpha,
                        lambda = M0.lambda
                        )
proc.time() - ptm
M0.coef = coef(M0.final.model)
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name       Beta
# 1     DER_deltar_tau_lep  1.1652586
# 2 DER_lep_eta_centrality  1.0132337
# 3   DER_pt_ratio_lep_tau -0.5674715
# 4            PRI_jet_num -0.4089144
# 5 DER_met_phi_centrality  0.3010004
# 6   DER_deltaeta_jet_jet -0.1939249