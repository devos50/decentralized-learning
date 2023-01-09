library(ggplot2)
library(dplyr)
library(gridExtra)

dat <- read.csv("../data/varying_alpha_accuracies.csv")
dat$alpha <- as.factor(dat$alpha)

p <- ggplot(dat, aes(x=round, y=accuracy, group=alpha, colour=alpha)) +
      geom_line() +
      theme_bw() +
      theme(legend.position=c(0.7, 0.45), legend.box.background = element_rect(colour = "black")) +
      guides(col = guide_legend(ncol = 2)) +
      xlab("Round") +
      ylab("Test Accuracy")

ggsave("varying_alpha.pdf", p, width=4, height=2.3)
