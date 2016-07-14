setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M8: 1 jet, no candidate mass estimate.
#     unexpected topology made it impossible to use probabalistic phase space
#     integration to estimate mass of Higgs boson candidate.
#     1 jet recorded, so all of the jet-pair-related metrics are undefined.
#     Removed PRI_jet_num because it is constant at 1 for this missingness category.
#     Therefore it both provides no predictive value and cannot be scaled.
#     21 independent variables.
# [1] "DER_mass_MMC": The estimated mass mH of the Higgs boson candidate, obtained
#     through a probabilistic phase space integration (may be undefined if the
#     topology of the event is too far from the expected topology)
#     The neutrinos are not measured in the detector, so their presence in the
#     final state makes it difficult to evaluate the mass of the Higgs candidate
#     on an event-by-event basis.
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
x = model.matrix(Label ~ ., M8.training[, -c(1, ncol(M8.training) - 1)])[, -1]
y = M8.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M8.lasso.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 1,
                         lambda = seq(from = 10^0,
                                      to = 10^-5,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M8.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M8 Lasso Regression")
# performing ridge regression to further hone lambda range.
ptm <- proc.time()
M8.ridge.models = glmnet(x = x,
                         y = y,
                         family = 'binomial',
                         alpha = 0,
                         lambda = seq(from = 10^1,
                                      to = 10^-4,
                                      length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M8.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M8 Ridge Regression")
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M8.best.model <- train(x = M8.training[, !names(M8.training) %in% c('EventId', 'Weight', 'Label')],
                       y = M8.training[, 'Label'],
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
# tuning parameters were (alpha = 0.8421053, lambda = 0.00078476), which resulted in
# a cross-validation accuracy of 0.9161616.
M8.alpha = M8.best.model$bestTune$alpha
M8.lambda = M8.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M8.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M8.alpha,
                        lambda = M8.lambda
                        )
proc.time() - ptm
M8.coef = coef(M8.final.model)
plot(2:nrow(M8.coef), abs(as.vector(M8.coef))[-1])
# empirical beta threshold = 0.1. Beta's above this are, to me, significant.
# There are three beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.1
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name       Beta
# 1     DER_deltar_tau_lep  0.9304610
# 2   DER_pt_ratio_lep_tau -0.7845409
# 3 DER_met_phi_centrality -0.1981846