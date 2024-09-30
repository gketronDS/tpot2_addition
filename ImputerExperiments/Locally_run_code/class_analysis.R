#install.packages("tidyverse")
#install.packages(c("agridat", "ggplot2", "ghibli", "ggdist"))
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

class <- read.csv("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/c/class_kw_test.csv")

myplot <- ggbetweenstats(
    data = class,
    #x = f1,
    #x = RMSEAcc,
    x = training_duration,
    #x = logloss,
    #x = balanced_accuracy,
    #x = accuracy,
    #x = auroc,
    y = Value,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/c/Saved_Analysis_myplot.png")