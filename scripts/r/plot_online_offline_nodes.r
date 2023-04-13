library(ggplot2)
library(dplyr)

dat <- read.csv("../data/n_100_cifar10/activities.csv")
dat$time <- dat$time / 3600

p <- ggplot(dat, aes(x=time, y=online)) +
     geom_line() +
     theme_bw() +
     xlab("Time (hours)") +
     ylab("Online Participants") +
     ylim(0, 1000) +
     geom_hline(yintercept=1000, linetype='dotted', col = 'red')

ggsave("../data/n_100_cifar10/activity.pdf", p, width=5, height=2.5)
