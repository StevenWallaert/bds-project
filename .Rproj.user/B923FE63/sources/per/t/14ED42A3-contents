library(tidyverse)
# binary - ae
Z <- read_csv("codings_tsne.csv", col_names = c("Z1", "Z2"))
y <- read_csv("y_codings.csv", col_names = "y")
yb <- ifelse(y$y, "normal", "attack")


ggplot(Z, aes(Z1, Z2, color=yb, shape=yb)) +
  geom_point(alpha=0.4) +
  labs(x="First t-SNE component",
       y="Second t-SNE component",
       color="Class",
       shape="Class") +
  theme_bw() +
  scale_color_brewer(palette = "Dark2")
# perplexity 30, all default

# coarse classification - ae
Z <- read_csv("codings_tsne.csv", col_names = c("Z1", "Z2"))
y <- read_csv("y_c.csv")

yc <- y$ctarget
ggplot(Z, aes(Z1, Z2, color=yc, shape=yc)) +
  geom_point(alpha=0.4) +
  labs(x="First t-SNE component",
       y="Second t-SNE component",
       color="Class",
       shape="Class") +
  theme_bw() +
  scale_color_brewer(palette = "Dark2")
