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

library(dplyr)

# Divide up data by type of missingness: 6 types.

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
# provides predictive value and cannot be scaled.


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