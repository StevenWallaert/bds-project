library(tidyverse)
res <- read_csv("results.csv")

res <- res %>%
  pivot_longer(Accuracy:AUC, names_to = "Metric", values_to = "Performance") %>%
  unite(Model, DimensionReduction, col = "Model", sep = "-") 

res %>%
  filter(Task == "Binary") %>%
  ggplot(aes(Model, Performance, fill=Metric)) +
  geom_col(position = "dodge") +
  coord_flip(ylim = c(0.97,1)) +
  scale_fill_brewer(palette = "Set1")

