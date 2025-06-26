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

ori <- read.csv("/Users/gabrielketron/class_kw/reg_wilcox_test_mnar.csv")
impute <- read.csv("/Users/gabrielketron/class_kw/reg_wilcox_test_mnar.csv")
#rint(colnames(reg))
#reg <- reg[reg$Model %in% c("Complex","Simple","NonSimple"),]
#reg <- reg[reg$r2>-1,]
#reg <- reg[reg$rmse<=20000,]
#,"NoImpute"
#print(ori)

newori <- data.frame(rmse=ori$ori)
#print(newori)
newori['Model'] <- 'Original'
#print(ori)
newimpute <- data.frame(rmse=impute$impute)
#colnames(ori)<-'rmse'
newimpute['Model'] <- 'Impute'

reg <- rbind(newori, newimpute)

print(reg)


#predrankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID,complexf1 = subset(reg,reg$Model %in% c("Mixed"))$rmse, simplef1 = subset(reg,reg$Model %in% c("Simple"))$rmse, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$rmse))
#reg[reg=="Impute_F"] <- "Impute with Complex Models before TPOT"
#reg[reg=="Complex"] <- "Mixed"
#reg[reg=="Simple"] #<- "Simple Models Only in TPOT"
#reg[reg=="NonSimple"] #<- "Complex Models Only in TPOT"
#reg[reg=="NoImpute"] <- "No Imputation in TPOT"

myplot <- ggwithinstats(
    data = reg,
    x = Model,
    #y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    #y = explained_var,
    y=rmse,
    ylab = "RMSE",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all", 
    title = "Wilcoxon Pairwise Comparison of Regression RMSE Between Treatments"
)

#predrankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID,complexf1 = subset(reg,reg$Model %in% c("Mixed"))$rmse, simplef1 = subset(reg,reg$Model %in% c("Simple"))$rmse, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$rmse))
#rint(predrankings)

#ggsave("/Users/gabrielketron/class_kw/wilcoxonreg_rmse_mnar.png")


myplot <- ggwithinstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    y = training_duration,
    #y = r2,
    #y = explained_var,
    ylab = "Training Duration (s)",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Training Time Between Treatments"
)

#trainingrankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID,complexf1 = subset(reg,reg$Model %in% c("Mixed"))$training_duration, simplef1 = subset(reg,reg$Model %in% c("Simple"))$training_duration, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$training_duration))
#rint(predrankings)

ggsave("/Users/gabrielketron/class_kw/ftreg_training_noimp_mnar.png")

myplot <- ggwithinstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    #y = training_duration,
    y = r2,
    #y = explained_var,
    ylab = "R2",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of R-Squared Between Treatments"
)

#r2rankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID,complexf1 = subset(reg,reg$Model %in% c("Mixed"))$r2, simplef1 = subset(reg,reg$Model %in% c("Simple"))$r2, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$r2))

#ggsave("/Users/gabrielketron/class_kw/ftreg_r2_mnar.png")

myplot <- ggbetweenstats(
    data = reg,
    x = Model,
    #y = rmse,
    #y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    y = explained_var,
    ylab = "Explained Variance",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Explained Variance Between Treatments"
)

#explainedrankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID,complexf1 = subset(reg,reg$Model %in% c("Mixed"))$explained_var, simplef1 = subset(reg,reg$Model %in% c("Simple"))$explained_var, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$explained_var))

#ggsave("/Users/gabrielketron/class_kw/ftreg_explainedvar_mnar.png")

#reg <- reg[reg$Model %in% c("Mixed Models in TPOT","Simple Models Only in TPOT","Complex Models Only in TPOT"),]

myplot <- ggwithinstats(
    data = reg,
    x = Model, 
    #y = rmse,
    y = RMSEAcc,
    #y = training_duration,
    #y = r2,
    #y = explained_var,
    ylab="Imputation RMSE",
    type = "nonparametric",
    #effsize.type = "d",
    p.adjust.method = "bonferroni",
    pairwise.display = "all",
    title = "Friedman Test Pairwise Comparison of Imputation RMSE Between Treatments"
)

#ggsave("/Users/gabrielketron/class_kw/ftreg_RMSEAcc_mnar.png")

#RMSErankings <- na.omit(data.frame(id = subset(reg,reg$Model %in% c("Mixed"))$ID, complexf1 = subset(reg,reg$Model %in% c("Mixed"))$RMSEAcc, simplef1 = subset(reg,reg$Model %in% c("Simple"))$RMSEAcc, mixedf1 = subset(reg,reg$Model %in% c("NonSimple"))$RMSEAcc))

friedprintout <- function(pairings, higher=FALSE){
    pairingswithoutid <- data.matrix(pairings[,-1])
    newout <- data.frame(id = unique(pairings$id))
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
    result <- data.frame(idcounter=idcounter, pair=pair, rankingmixed=rankingmixed, rankingsimple=rankingsimple, rankingnonsimple=rankingnonsimple)
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
    #print(length(idcounter))
    #print(length(rankingmixed))
    #print(length(rankingsimple))
    #print(length(rankingnonsimple))
    #print(length(pair))
    result[nrow(result)+1,] = c(idcounter[length(idcounter)],pair[length(pair)],
    rankingmixed[length(rankingmixed)], rankingsimple[length(rankingsimple)],
    rankingnonsimple[length(rankingnonsimple)])
    print(result)
    
}
sink("/Users/gabrielketron/class_kw/ftreg_pred_results.txt")
friedprintout(predrankings)
sink()

#sink("/Users/gabrielketron/class_kw/ftreg_train_results.txt")
#friedprintout(trainingrankings)
#sink()

#sink("/Users/gabrielketron/class_kw/ftreg_r2_results.txt")
#friedprintout(r2rankings, higher = TRUE)
#sink()

#ink("/Users/gabrielketron/class_kw/ftreg_explained_results.txt")
#friedprintout(explainedrankings, higher = TRUE)
#sink()

#sink("/Users/gabrielketron/class_kw/ftreg_imp_results.txt")
#friedprintout(RMSErankings)
#sink()