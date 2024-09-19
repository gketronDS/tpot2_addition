#install.packages("tidyverse")
i#nstall.packages(c("agridat", "ggplot2", "ghibli", "ggdist"))
library(agridat)
library(ggplot2)
library(ghibli)
library(ggdist)
library(tidyverse)
#install.packages("ggstatsplot")
library(ggstatsplot)
#install.packages("ISLR")
library(ISLR)

set.seed(1)

reg <- read.csv("tpot2/ImputerExperiments/data/r/reg_kw_test.csv")

myplot <- ggbetweenstats(
    data = reg,
    x = RMSEAcc,
    y = value,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("tpot2/ImputerExperiments/data/r/Saved_Analysis_myplot.png")