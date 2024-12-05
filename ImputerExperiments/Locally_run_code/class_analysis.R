#!/usr/bin/env Rscript

#install.packages("tidyverse")
#install.packages(c("tidyverse","agridat", "ggplot2", "ghibli", "ggdist", "ggstatsplot", "ISLR"))
#library(tidyverse, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(agridat, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ggplot2, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ghibli, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ggstatsplot, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ISLR, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(tidyverse, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")

library(ggplot2)
library(ghibli)
library(ggdist)
library(tidyverse)
#install.packages("ggstatsplot")
library(ggstatsplot)
#install.packages("ISLR")
library(ISLR)

set.seed(1)

class <- read.csv("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/class_kw_test.csv")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_f1.png")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    #y = f1,
    y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_RMSEAcc.png")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_logloss.png")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)

ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_bal_acc.png")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    y = accuracy,
    #y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_acc.png")

myplot <- ggbetweenstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    y = auroc,
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all"
)


ggsave("/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/kwclass_auroc.png")
