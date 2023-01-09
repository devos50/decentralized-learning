library(ggplot2)
library(dplyr)
library(gridExtra)

dat <- read.csv("../data/varying_temperature_accuracies.csv")
dat$temperature <- as.factor(dat$temperature)

p <- ggplot(dat, aes(x=round, y=accuracy, group=temperature, colour=temperature)) +
      geom_line() +
      theme_bw() +
      theme(legend.position=c(0.7, 0.45), legend.box.background = element_rect(colour = "black")) +
      guides(col = guide_legend(ncol = 2)) +
      xlab("Round") +
      ylab("Test Accuracy")

ggsave("varying_temperature.pdf", p, width=4, height=2.3)
