setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
load('kaggle.RData')

library(dplyr)
library(glmnet)
library(caret)

# M11: 0 jets, no candidate mass estimate.
#      No jets recorded, so all of the continous jet metrics are undefined.
#      Removed PRI_jet_num and PRI_jet_all_pt because both are constant at 0
#      for this missingness category. Therefore neither provides predictive
#      value and cannot be scaled.
#      17 independent variables.
# [1] "DER_mass_MMC": The estimated mass mH of the Higgs boson candidate, obtained
#     through a probabilistic phase space integration (may be undefined if the
#     topology of the event is too far from the expected topology)
#     The neutrinos are not measured in the detector, so their presence in the
#     final state makes it difficult to evaluate the mass of the Higgs candidate
#     on an event-by-event basis.
# [2] "DER_deltaeta_jet_jet": The absolute value of the pseudorapidity
#       separation (22) between the two jets (undefined if PRI jet num ≤ 1).
# [3] "DER_mass_jet_jet": The invariant mass (20) of the two jets (undefined if
#       PRI jet num ≤ 1).
# [4] "DER_prodeta_jet_jet": The product of the pseudorapidities of the two jets
#       (undefined if PRI jet num ≤ 1).
# [5] "DER_lep_eta_centrality": The centrality of the pseudorapidity of the
#       lepton w.r.t. the two jets (undefined if PRI jet num ≤ 1) where ηlep is
#       the pseudorapidity of the lepton and η1 and η2 are the pseudorapidities
#       of the two jets. The centrality is 1 when the lepton is on the bisector
#       of the two jets, decreases to 1/e when it is collinear to one of the
#       jets, and decreases further to zero at infinity.
# [6] "PRI_jet_num": The number of jets (integer with value of 0, 1, 2 or 3;
#       possible larger values have been capped at 3).
# [7] "PRI_jet_leading_pt": The transverse momentum  p2x + p2y of the leading
#       jet, that is the jet with largest transverse momentum (undefined if PRI
#       jet num = 0).
# [8] "PRI_jet_leading_eta": The pseudorapidity η of the leading jet (undefined
#       if PRI jet num = 0).
# [9] "PRI_jet_leading_phi": The azimuth angle φ of the leading jet (undefined
#       if PRI jet num = 0).
# [10] "PRI_jet_subleading_pt": The transverse momentum  p2x + p2y of the
#         leading jet, that is, the jet with second largest transverse momentum
#         (undefined if PRI jet num ≤ 1).
# [11] "PRI_jet_subleading_eta": The pseudorapidity η of the subleading jet
#         (undefined if PRI jet num ≤ 1).
# [12] "PRI_jet_subleading_phi": The azimuth angle φ of the subleading jet
#         (undefined if PRI jet num ≤ 1).
# [13] "PRI_jet_all_pt": The scalar sum of the transverse momentum of all the
#         jets of the events.

#------------------------------------------------------------------------------
# Elasticnet Regularization
#------------------------------------------------------------------------------
elasticnet.trainControl = trainControl(method = 'cv',
                                       number = 10
                                       )
#preparing data for glmnet-facilitated lasso regression
x = model.matrix(Label ~ ., M11.training[, -c(1, ncol(M11.training) - 1)])[, -1]
y = M11.training$Label
#Alpha = 1 for lasso regression. Getting a range for lambda.
ptm <- proc.time()
M11.lasso.models = glmnet(x = x,
                          y = y,
                          family = 'binomial',
                          alpha = 1,
                          lambda = seq(from = 10^0,
                                       to = 10^-5,
                                       length.out = 100)
)
proc.time() - ptm
#Visualizing the lasso regression shrinkage to verify lambda range.
plot(M11.lasso.models,
     xvar = "lambda",
     label = TRUE,
     main = "M11 Lasso Regression")
# performing ridge regression to further hone lambda range.
ptm <- proc.time()
M11.ridge.models = glmnet(x = x,
                          y = y,
                          family = 'binomial',
                          alpha = 0,
                          lambda = seq(from = 10^1,
                                       to = 10^-4,
                                       length.out = 100)
)
proc.time() - ptm
#Visualizing the ridge regression shrinkage to verify lambda range.
plot(M11.ridge.models,
     xvar = "lambda",
     label = TRUE,
     main = "M11 Ridge Regression")
#Running 10-fold cross validation to determine best (alpha, lambda) pair
set.seed(0)
ptm <- proc.time()
M11.best.model <- train(x = M11.training[, !names(M11.training) %in% c('EventId', 'Weight', 'Label')],
                        y = M11.training[, 'Label'],
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
# This took five minutes to run. With 100 (alpha, lambda) combinations, the best
# tuning parameters were (alpha = 0.3684211, lambda = 0.000379269), which resulted in
# a cross-validation accuracy of 0.9480917.
M11.alpha = M11.best.model$bestTune$alpha
M11.lambda = M11.best.model$bestTune$lambda

# Now, I want to use these parameters on all the training data to get the
# coefficients. I'll try that now.

ptm <- proc.time()
M11.final.model = glmnet(x = x,
                        y = y,
                        family = 'binomial',
                        alpha = M11.alpha,
                        lambda = M11.lambda
                        )
proc.time() - ptm
M11.coef = coef(M11.final.model)
plot(2:nrow(M11.coef), abs(as.vector(M11.coef))[-1])
# empirical beta threshold = 0.2 Beta's above this are, to me, significant.
# There are three beta coefficients that have magnitudes above this threshold.
# I will extract these from the coefficient matrix:
beta_threshold = 0.2
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
# #Now, the variables that came up as significant for the full data are, in
# #descending order by coefficient magnitude:
# Variable_Name       Beta
# 1     DER_deltar_tau_lep  1.8328526
# 2   DER_pt_ratio_lep_tau -1.2149866
# 3 DER_met_phi_centrality -0.2346662