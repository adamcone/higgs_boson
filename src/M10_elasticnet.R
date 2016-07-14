setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M10: 0 jets, candidate mass estimate made.
#      No jets recorded, so all of the continous jet metrics are undefined.
#      Removed PRI_jet_num and PRI_jet_all_pt because both are constant at 0
#      for this missingness category. Therefore neither provides predictive
#      value and cannot be scaled.
#      18 independent variables.
# [1] "DER_deltaeta_jet_jet": The absolute value of the pseudorapidity
#       separation (22) between the two jets (undefined if PRI jet num ≤ 1).
# [2] "DER_mass_jet_jet": The invariant mass (20) of the two jets (undefined if
#       PRI jet num ≤ 1).
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
# [6] "PRI_jet_leading_pt": The transverse momentum  p2x + p2y of the leading
#       jet, that is the jet with largest transverse momentum (undefined if PRI
#       jet num = 0).
# [7] "PRI_jet_leading_eta": The pseudorapidity η of the leading jet (undefined
#       if PRI jet num = 0).
# [8] "PRI_jet_leading_phi": The azimuth angle φ of the leading jet (undefined
#       if PRI jet num = 0).
# [9] "PRI_jet_subleading_pt": The transverse momentum  p2x + p2y of the
#       leading jet, that is, the jet with second largest transverse momentum
#       (undefined if PRI jet num ≤ 1).
# [10] "PRI_jet_subleading_eta": The pseudorapidity η of the subleading jet
#       (undefined if PRI jet num ≤ 1).
# [11] "PRI_jet_subleading_phi": The azimuth angle φ of the subleading jet
#       (undefined if PRI jet num ≤ 1).
# [12] "PRI_jet_all_pt": The scalar sum of the transverse momentum of all the
#       jets of the events.
#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
elasticnet.trainControl = trainControl(method = 'cv',
                                       number = 10
                                       )

#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M10.training[, -c(1, ncol(M10.training) - 1)])[, -1]
y = M10.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M10.lasso.models = glmnet(x = x,
                          y = y,
                          family = 'binomial',
                          alpha = 1,
                          lambda = seq(from = 10^0,
                                       to = 10^-5,
                                       length.out = 100)
)
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M10.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M10 Lasso Regression")
# performing ridge regression to further hone lambda range.
ptm <- proc.time()
M10.ridge.models = glmnet(x = x,
                          y = y,
                          family = 'binomial',
                          alpha = 0,
                          lambda = seq(from = 10^1,
                                       to = 10^-4,
                                       length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M10.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M10 Ridge Regression")
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M10.best.model <- train(x = M10.training[, !names(M10.training) %in% c('EventId', 'Weight', 'Label')],
                        y = M10.training[, 'Label'],
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
# tuning parameters were (alpha = 0.7777778, lambda = 0.001), which resulted in
# a cross-validation accuracy of 0.7882099.
M10.alpha = M10.best.model$bestTune$alpha
M10.lambda = M10.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M10.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M10.alpha,
                        lambda = M10.lambda
                        )
proc.time() - ptm
M10.coef = coef(M10.final.model)
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name      Beta
# 1   DER_deltar_tau_lep  2.909918
# 2 DER_pt_ratio_lep_tau -1.543277