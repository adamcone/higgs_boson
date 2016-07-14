setwd('/Users/adamcone/Desktop/projects/Kaggle/code')
library(dplyr)

s.prob.reduced = read.table(file = 'S.Prob.Reduce.csv',
                            header = TRUE,
                            sep = ',',
                            stringsAsFactors = FALSE
                            )

# get rid of extra column
s.prob.reduced = select(s.prob.reduced, -1)

# replace S.Probability column with RankOrder column for submission

Reduced.Final = arrange(s.prob.reduced, by = S.Probability) %>%
                           mutate(., RankOrder = seq(1, nrow(s.prob.reduced))) %>%
                           mutate(., Class = ifelse(S.Probability > 0.31, 's', 'b')) %>%
                           select(., c(EventId,
                                       RankOrder,
                                       Class)
                                  ) %>%
                           arrange(., by = EventId)

write.csv(x = Reduced.Final,
          file = 'Reduced_Final.csv',
          row.names = FALSE
          )

#---------------

s.prob.full = read.table(file = 'S.Prob.Full.csv',
                         header = TRUE,
                         sep = ',',
                         stringsAsFactors = FALSE
                         )

# replace S.Probability column with RankOrder column for submission

Full.Final = arrange(s.prob.full, by = S.Probability) %>%
  mutate(., RankOrder = seq(1, nrow(s.prob.full))) %>%
  mutate(., Class = ifelse(S.Probability > 0.71, 's', 'b')) %>%
  select(., c(EventId,
              RankOrder,
              Class)
         ) %>%
  arrange(., by = EventId)

write.csv(x = Full.Final,
          file = 'Full_Final.csv',
          row.names = FALSE
)