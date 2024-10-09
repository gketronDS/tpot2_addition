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

reg <- read.csv("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/reg_kw_test.csv")

myplot <- ggbetweenstats(
    data = reg,
    x = Model,
    #y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    #y = explained_var,
    y=rmse,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/kwreg_rmse.png")

myplot <- ggbetweenstats(
    data = reg,
    x = Model, 
    #y = rmse,
    y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    #y = explained_var,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/kwreg_RMSEAcc.png")

myplot <- ggbetweenstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    y = training_duration,
    #y = r2,
    #y = explained_var,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/kwreg_training.png")

myplot <- ggbetweenstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    #y = training_duration,
    y = r2,
    #y = explained_var,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/kwreg_r2.png")

myplot <- ggbetweenstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    y = explained_var,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/kwreg_explainedvar.png")