setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M1: no candidate mass assessment. 29 independent variables.
# [1] "DER_mass_MMC": The estimated mass mH of the Higgs boson candidate, obtained
#     through a probabilistic phase space integration (may be undefined if the
#     topology of the event is too far from the expected topology)
#     The neutrinos are not measured in the detector, so their presence in the
#     final state makes it difficult to evaluate the mass of the Higgs candidate
#     on an event-by-event basis.


#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
elasticnet.trainControl = trainControl(method = 'cv',
                                       number = 10
                                       )
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M1.training[, -c(1, ncol(M1.training) - 1)])[, -1]
y = M1.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M1.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = seq(from = 10^0,
                                      to = 10^-5,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M1.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M1 Lasso Regression")
# performing ridge regression to further hone lambda range.
ptm <- proc.time()
M1.ridge.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 0,
                         lambda = seq(from = 10^1,
                                      to = 10^-4,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M1.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M1 Ridge Regression")
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M1.best.model <- train(x = M1.training[, !names(M1.training) %in% c('EventId', 'Weight', 'Label')],
                       y = M1.training[, 'Label'],
                       method='glmnet',
                       metric = "Accuracy",
                       preProc = c('center', 'scale'),
                       tuneGrid = expand.grid(.alpha=seq(from = 0,
                                                         to = 1,
                                                         length.out = 20),
                                              .lambda = 10^seq(from = -5,
                                                               to = 1,
                                                               length.out = 20)
                       ),
                       trControl = elasticnet.trainControl
                       )
proc.time() - ptm
# This took a minute to run. With 400 (alpha, lambda) combinations, the best
# tuning parameters were (alpha = 0.3684211, lambda = 0.001623777), which resulted in
# a cross-validation accuracy of 0.9092300.
M1.alpha = M1.best.model$bestTune$alpha
M1.lambda = M1.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M1.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M1.alpha,
                        lambda = M1.lambda
                        )
proc.time() - ptm
M1.coef = coef(M1.final.model)
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name       Beta
# 1            PRI_jet_num -0.8894924
# 2 DER_lep_eta_centrality  0.4443602
# 3     DER_deltar_tau_lep  0.4101015
# 4   DER_pt_ratio_lep_tau -0.3796870