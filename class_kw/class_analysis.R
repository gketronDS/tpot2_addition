#install.packages("tidyverse")

#install.packages(c("dplyr","tidyverse", "ggplot2", "ggdist", "ggstatsplot", "effectsize","PMCMRplus", "PMCMR", "stats"))

#/home/ketrong/R/x86_64-pc-linux-gnu-library/4.4/
#library(tidyverse, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(agridat, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ggplot2, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ghibli, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ggstatsplot, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(ISLR, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")
#library(tidyverse, lib.loc = "/R/x86_64-pc-linux-gnu-library/4.4")

library(ggplot2)
#library(ghibli)
library(ggdist)
library(tidyverse)
#install.packages("ggstatsplot")
library(ggstatsplot)
#install.packages("ISLR")
#library(ISLR)
library(dplyr)
library(PMCMRplus)
library(PMCMR)
library(effectsize)
library(stats)
set.seed(1)

class <- read.csv("/Users/gabrielketron/class_kw/class_kw_test_mnar.csv")

#print(colnames(class))
class <- class[class$Model %in% c("Complex","Simple","NonSimple"),]
#,"NoImpute"
#reg[reg=="Impute_F"] <- "Impute with Complex Models before TPOT"
class[class=="Complex"] <- 'Mixed'
class[class=="Simple"] 
class[class=="NonSimple"] 
#class[class=="NoImpute"] <- "No Imputation"


myplot <- ggwithinstats(
    data = class,
    x = Model,
    y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    #y = training_duration,
    ylab = "F1 Macro",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of F1-Macro Score Between Treatments"
)

f1rankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$f1, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$f1, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$f1))

#print(f1rankings)


ggsave("/Users/gabrielketron/class_kw/ftclass_f1_check_mnar.png")



myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    ylab = "Log Loss",
    type = "nonparametric",
    effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Log Loss Between Treatments"
)

loglossrankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$logloss, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$logloss, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$logloss))
#ggsave("/Users/gabrielketron/class_kw/ftclass_logloss_mnar.png")

myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    ylab = "Balanced Accuracy",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Balanced Accuracy Between Treatments"
)
balanced_accuracyrankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$balanced_accuracy, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$balanced_accuracy, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$balanced_accuracy))
#ggsave("/Users/gabrielketron/class_kw/ftclass_bal_acc_mnar.png")

myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    y = accuracy,
    #y = auroc,
    ylab = "Accuracy",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Accuracy Between Treatments"
)

accuracyrankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$accuracy, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$accuracy, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$accuracy))

#ggsave("/Users/gabrielketron/class_kw/ftclass_acc_mnar.png")

myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    y = auroc,
    ylab = "AUROC",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of AUROC Between Treatments"
)

aurocrankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$auroc, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$auroc, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$auroc))

#ggsave("/Users/gabrielketron/class_kw/ftclass_auroc_mnar.png")

myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    #y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    y = training_duration,
    ylab = "Training Duration",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Training Time Between Treatments"
)

trainingrankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, 
    complexf1 = subset(class,class$Model %in% c("Mixed"))$training_duration, 
    simplef1 = subset(class,class$Model %in% c("Simple"))$training_duration, 
    mixedf1 = subset(class,class$Model %in% c("NonSimple"))$training_duration))

ggsave("/Users/gabrielketron/class_kw/ftclass_duration_noimp_mnar.png")
#print(class)
#class <- class[class$Model %in% c("Mixed Models in TPOT","Simple Models Only in TPOT","Complex Models Only in TPOT"),]

myplot <- ggwithinstats(
    data = class,
    x = Model,
    #y = f1,
    y = RMSEAcc,
    #y = logloss,
    #y = balanced_accuracy,
    #y = accuracy,
    #y = auroc,
    ylab = "Imputation RMSE",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Imputation RMSE Between Treatments"
)

ggsave("/Users/gabrielketron/class_kw/ftclass_RMSEAcc_noimp_mnar.png")

RMSErankings <- na.omit(data.frame(id = subset(class,class$Model %in% c("Mixed"))$ID, complexf1 = subset(class,class$Model %in% c("Mixed"))$RMSEAcc, simplef1 = subset(class,class$Model %in% c("Simple"))$RMSEAcc, mixedf1 = subset(class,class$Model %in% c("NonSimple"))$RMSEAcc))


friedprintout <- function(pairings, higher=FALSE){
    pairingswithoutid <- data.matrix(pairings[,-1])
    newout <- data.frame(id = unique(pairings$id))
    print(length(newout$id))
    pair <- vector()
    rankingmixed <- vector()
    rankingsimple <- vector()
    rankingnonsimple <- vector()
    idcounter <- vector()
    vectout <- as.vector(newout)
    for (i in vectout$id){
        #print(i)
        idcounter[length(idcounter)+1] <- i
        sectional <- subset(pairings, id == i)
        #print(sectional)
        sectional <- sectional[,-1]
        if (higher){
            #print(higher)
            sectional_neg <- sectional 
            sectional_neg[sapply(sectional_neg, is.numeric)] <- sectional_neg[sapply(sectional_neg, is.numeric)]*-1
            #sectional <- sectional_neg
            #print(sectional_neg)
            new_sectional <- sectional_neg
            myData2 <- data.frame(new_sectional, t(apply(new_sectional, 1, rank, ties.method='average')))
        }
        else{
            new_sectional <- sectional
            myData2 <- data.frame(new_sectional, t(apply(new_sectional, 1, rank, ties.method='average')))
        }
        myTable <- myData2[,4:6]
        colnames(myTable)<-c("Mixed", "Simple", "NonSimple")
        R <- colSums(myTable)
        
        R <- data.frame(t(R))
        #print(R)
        
        rankingmixed[length(rankingmixed)+1] <- R$Mixed
        rankingsimple[length(rankingsimple)+1] <- R$Simple
        rankingnonsimple[length(rankingnonsimple)+1] <- R$NonSimple

        smallest <- colnames(R)[max.col(-R)]
        new_sectional <- data.matrix(new_sectional)
        ftest <- friedmanTest(new_sectional)
        contest <- frdAllPairsConoverTest(y=new_sectional, p.adjust.method="bonferroni")
        if (ftest$p.value > 0.05){
            pair[length(pair)+1] <- "No Difference"
        }
        else{
            if (smallest == "NonSimple"){
                if ((contest$p.value[2,1] <= 0.05) & (contest$p.value[2,2]<= 0.05)){
                    pair[length(pair)+1] <- smallest
                }
                else if (contest$p.value[2,1] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and Simple')
                }
                else if (contest$p.value[2,2] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and Mixed')
                }
                else{
                pair[length(pair)+1] <- "No Difference"
                }
            }
            else if (smallest == "Simple"){
                if ((contest$p.value[1,1] <= 0.05) & (contest$p.value[2,2]<= 0.05)){
                    pair[length(pair)+1] <- smallest
                }
                else if (contest$p.value[1,1] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and NonSimple')
                }
                else if (contest$p.value[2,2] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and Mixed')
                }
                else{
                pair[length(pair)+1] <- "No Difference"
                }
            }
            else if (smallest == "Mixed"){
                if ((contest$p.value[1,1] <= 0.05) & (contest$p.value[2,1]<= 0.05)){
                    pair[length(pair)+1] <- smallest
                }
                else if (contest$p.value[1,1] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and NonSimple')
                }
                else if (contest$p.value[2,1] <= 0.05){
                    pair[length(pair)+1] <- paste(smallest,'and Simple')
                }
                else{
                pair[length(pair)+1] <- "No Difference"
                }
            }
            else{
                pair[length(pair)+1] <- "No Difference"
            }
        }
        if (length(pair) != length(idcounter)){
            print('Error on')
            print(i)
        }
    }
    result <- data.frame(idcounter=idcounter, pair=pair,rankingmixed=rankingmixed, rankingsimple=rankingsimple, rankingnonsimple=rankingnonsimple)
    result <- result[order(result$idcounter, decreasing = FALSE), ]
    idcounter[length(idcounter)+1] <- "Total" 
    if (higher){
        #print(higher)
        sectional_neg <- pairingswithoutid
        sectional_neg[sapply(sectional_neg, is.numeric)] <- sectional_neg[sapply(sectional_neg, is.numeric)]*-1
        new_sectional <- sectional_neg
        #print(sectional_neg)
        myDatatotal <- data.frame(new_sectional, t(apply(new_sectional, 1, rank, ties.method='average')))
    }
    else{
        new_sectional <- pairingswithoutid
        myDatatotal <- data.frame(new_sectional, t(apply(new_sectional, 1, rank, ties.method='average')))
    }
    #print(myDatatotal)
    myTabletotal <- myDatatotal[,4:6]
    colnames(myTabletotal)<-c("Mixed", "Simple", "NonSimple")
    R <- colSums(myTabletotal) 
    R <- data.frame(t(R))
    rankingmixed[length(rankingmixed)+1] <- R$Mixed
    rankingsimple[length(rankingsimple)+1] <- R$Simple
    rankingnonsimple[length(rankingnonsimple)+1] <- R$NonSimple
    smallest <- colnames(R)[max.col(-R)]
    new_sectional <- data.matrix(new_sectional)
    tested <- friedmanTest(new_sectional)
    #print(tested$p.value)
    #print(tested$statistic)
    out <- frdAllPairsConoverTest(y=new_sectional, p.adjust.method="bonferroni")
    #print(out$p.value)
    #print(out$p.value[1,2])
    #print(out$statistic)
    if (tested$p.value > 0.05){
            pair[length(pair)+1] <- "No Difference"
        }
    else{
        if (smallest == "NonSimple"){
            if ((out$p.value[2,1] <= 0.05) & (out$p.value[2,2]<= 0.05)){
                pair[length(pair)+1] <- smallest
            }
            else if (out$p.value[2,1] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and Simple')
            }
            else if (out$p.value[2,2] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and Mixed')
            }
            else{
            pair[length(pair)+1] <- "No Difference"
            }
        }
        else if (smallest == "Simple"){
            if ((out$p.value[1,1] <= 0.05) & (out$p.value[2,2]<= 0.05)){
                pair[length(pair)+1] <- smallest
            }
            else if (out$p.value[1,1] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and NonSimple')
            }
            else if (out$p.value[2,2] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and Mixed')
            }
            else{
            pair[length(pair)+1] <- "No Difference"
            }
        }
        else if (smallest == "Mixed"){
            if ((out$p.value[1,1] <= 0.05) & (out$p.value[2,1]<= 0.05)){
                pair[length(pair)+1] <- smallest
            }
            else if (out$p.value[1,1] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and NonSimple')
            }
            else if (out$p.value[2,1] <= 0.05){
                pair[length(pair)+1] <- paste(smallest,'and Simple')
            }
            else{
            pair[length(pair)+1] <- "No Difference"
            }
        }
        else{
            pair[length(pair)+1] <- "No Difference"
        }
    }
    result[nrow(result)+1,] = c(idcounter[length(idcounter)], pair[length(pair)],
    rankingmixed[length(rankingmixed)], rankingsimple[length(rankingsimple)],
    rankingnonsimple[length(rankingnonsimple)])
    print(result)
}

#sink("/Users/gabrielketron/class_kw/ftclass_f1_results.txt")
#friedprintout(f1rankings, higher=TRUE)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_logloss_results.txt")
#friedprintout(loglossrankings)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_balacc_results.txt")
#friedprintout(balanced_accuracyrankings, higher=TRUE)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_acc_results.txt")
#friedprintout(accuracyrankings, higher=TRUE)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_auroc_results.txt")
#friedprintout(aurocrankings, higher=TRUE)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_training_results.txt")
#friedprintout(trainingrankings)
#sink()

#sink("/Users/gabrielketron/class_kw/ftclass_RMSE_results.txt")
#friedprintout(RMSErankings)
#sink()