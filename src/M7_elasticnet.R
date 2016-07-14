setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M7: 1 jet, candidate mass assessment made.
#     1 jet recorded, so all of the jet-pair-related metrics are undefined.
#     Removed PRI_jet_num because it is constant at 1 for this missingness category.
#     Therefore it both provides no predictive value and cannot be scaled.
#     22 independent variables.
# [1] "DER_deltaeta_jet_jet": The absolute value of the pseudorapidity separation
#       between the two jets (undefined if PRI jet num ≤ 1).
# [2] "DER_mass_jet_jet": The invariant mass of the two jets (undefined if PRI
#       jet num ≤ 1).
# [3] "DER_prodeta_jet_jet": The product of the pseudorapidities of the two jets
#       (undefined if PRI jet num ≤ 1).
# [4] "DER_lep_eta_centrality": The centrality of the pseudorapidity of the
#       lepton w.r.t. the two jets (undefined if PRI jet num ≤ 1) where ηlep is
#       the pseudorapidity of the lepton and η1 and η2 are the pseudorapidities
#       of the two jets. The centrality is 1 when the lepton is on the bisector
#       of the two jets, decreases to 1/e when it is collinear to one of the
#       jets, and decreases further to zero at infinity.
# [5] "PRI_jet_num": The number of jets (integer with value of 0, 1, 2 or 3;
#       possible larger values have been capped at 3).
# [6] "PRI_jet_subleading_pt": The transverse momentum  p2x + p2y of the leading
#       jet, that is, the jet with second largest transverse momentum (undefined
#       if PRI jet num ≤ 1).
# [7] "PRI_jet_subleading_eta": The pseudorapidity η of the subleading jet
#       (undefined if PRI jet num ≤ 1).
# [8] "PRI_jet_subleading_phi": The azimuth angle φ of the subleading jet
#       (undefined if PRI jet num ≤ 1).

#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
elasticnet.trainControl = trainControl(method = 'cv',
                                       number = 10
                                       )
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M7.training[, -c(1, ncol(M7.training) - 1)])[, -1]
y = M7.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M7.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = seq(from = 10^0,
                                      to = 10^-5,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M7.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M7 Lasso Regression")
# performing ridge regression to further hone lambda range.
ptm <- proc.time()
M7.ridge.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 0,
                         lambda = seq(from = 10^1,
                                      to = 10^-4,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M7.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M7 Ridge Regression")
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M7.best.model <- train(x = M7.training[, !names(M7.training) %in% c('EventId', 'Weight', 'Label')],
                       y = M7.training[, 'Label'],
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
# This took six minutes to run. With 100 (alpha, lambda) combinations, the best
# tuning parameters were (alpha = 1, lambda = 0.001), which resulted in
# a cross-validation accuracy of 0.6945213.
M7.alpha = M7.best.model$bestTune$alpha
M7.lambda = M7.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M7.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M7.alpha,
                        lambda = M7.lambda
                        )
proc.time() - ptm
M7.coef = coef(M7.final.model)
plot(2:nrow(M7.coef), abs(as.vector(M7.coef))[-1])
# empirical beta threshold = 0.1. Beta's above this are, to me, significant.
# There are four beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.1
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name       Beta
# 1     DER_deltar_tau_lep  1.2538340
# 2   DER_pt_ratio_lep_tau -0.7282635
# 3 DER_met_phi_centrality  0.2412309