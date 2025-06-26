#install.packages("tidyverse")
#install.packages(c("tidyverse", "ggplot2", "ggdist", "ggstatsplot", "cowplot", "dplyr", 'PupillometryR', 'Rlab'))
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
library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(Rlab)

spinefirstresults <- read.csv("/Users/gabrielketron/class_kw/spine_first.csv")
expertandcomplexresults <- read.csv("/Users/gabrielketron/class_kw/spine.csv")
simplefirstandsimpleresults <- read.csv("/Users/gabrielketron/class_kw/spine_simple.csv")
nonsimple<- read.csv("/Users/gabrielketron/class_kw/spine_nonsimple_new.csv")
SHAPE <- c(21, 21)
# permutation test with t-test statistic
# assuming we are using an alpha of 0.05
permutation_test <- function(x, y, seed, alternative, comp, metric) {
  # Set the random seed for reproducibility
  set.seed(seed)

  # Number of permutations
  n_permutations <- 100000

  # Calculate the observed difference in means
  observed_diff <- x - y
  
  critical_value <- sum(observed_diff)

  print(paste('percent difference:',critical_value*100/15))
  final_obs <- critical_value
  l <- 0
  u <- 0

  # Get number of samples
  n_x <- length(observed_diff)

  print(paste('critical_val:', critical_value))

  # Generate permutation differences
  permutation_diffs <- numeric(n_permutations)

  # Use a reproducible random sequence for each permutation
  # Generate unique seeds for each permutation
  seeds <- sample.int(1e9, n_permutations)

  for (i in 1:n_permutations) {
    # Set seed for this permutation
    set.seed(seeds[i])
    # Shuffle signs
    samplebern <- rbern(n_x, prob=0.5)
    signvector <- 1 - 2*samplebern
    # Get test statistic a
    permutation_diffs[i] <- sum(signvector*abs(observed_diff))
    if (permutation_diffs[i] <= critical_value) {
         l <- l + 1
    }
    if (permutation_diffs[i] >= critical_value) {
         u <- u + 1
    }
  }
  #print(signvector)
  pupper <- u/n_permutations
  plower <- l/n_permutations
  ptwo <- min(1, 2*pupper, 2*plower)

  #print(pupper)
  #print(plower)
  #print(ptwo)

  # sort permutation_diffs
  permutation_diffs <- sort(permutation_diffs)

  if (alternative == "l") {
    # is the observed difference < than the 5th percentile
    print(paste('permutation_diffs[0.05 * n_permutations]:',
            permutation_diffs[0.05 * n_permutations]))

    if (plower < 0.05) {
      print('reject null hypothesis')
    }
    else {
      print('fail to reject null hypothesis')
    }

    # if p_value is 0
    p_value <- plower
    print(paste('p-value:', p_value))
    

    # make histogram plot
    df <- data.frame(difference = permutation_diffs)
    df$category <- ifelse(df$difference < final_obs, 'not', 'extreme')

    plot <- ggplot(df, aes(x = difference, fill = category)) +
        geom_histogram(bins = 100,
                        color = "black",
                        alpha = 0.7) +
        geom_vline(xintercept = final_obs,
                    color = "red",
                    linetype = "dotted",
                    linewidth = 1
                    ) +
        labs(title = paste(comp,metric,"Permutation Test: Paired Sign Test Stat Differences"),
                x = paste("Sum of ",metric,"Differences"),
                y = "Frequency"
                ) +
        theme_classic() +
        scale_colour_manual(values = c('black', 'green')) +
        scale_fill_manual(values = c('black', 'green'))

  print(plot)

  } else if (alternative == "g") {
    # is the observed difference > than the 95th percentile
    print(paste('permutation_diffs[0.95 * n_permutations]:',
            permutation_diffs[0.95 * n_permutations]))

    if (0.95 < pupper) {
      print('reject null hypothesis')
    }
    else{
      print('fail to reject null hypothesis')
    }

    # if p_value is 0
    p_value <- pupper
    print(paste('p-value:', p_value))
    

    # make histogram plot
    df <- data.frame(difference = permutation_diffs)
    df$category <- ifelse(df$difference < final_obs, 'not', 'extreme')

    plot <- ggplot(df, aes(x = difference, fill = category)) +
        geom_histogram(bins = 100,
                        color = "black",
                        alpha = 0.7) +
        geom_vline(xintercept = final_obs,
                    color = "red",
                    linetype = "dotted",
                    linewidth = 1
                    ) +
        labs(title = paste(comp,metric,"Permutation Test: Paired Sign Test Stat Differences"),
                x = paste("Sum of ",metric,"Differences"),
                y = "Frequency"
                ) +
        theme_classic() +
        scale_shape_manual(values = SHAPE,) +
        scale_colour_manual(values = c('green', 'black')) +
        scale_fill_manual(values = c('green', 'black'))

  print(plot)
  } else if (alternative == "t") {
    # is the observed difference within 2.5th and 97.5th percentile
    lower <- final_obs < permutation_diffs[0.025 * n_permutations]
    print(paste('lower:', permutation_diffs[0.025 * n_permutations]))
    upper <- final_obs > permutation_diffs[0.975 * n_permutations]
    print(paste('upper:', permutation_diffs[0.975 * n_permutations]))

    if (ptwo < 0.05) {
      print('reject null hypothesis')
    }
    else{
      print('fail to reject null hypothesis')
    }

    # if p_value is 0
    p_value <- ptwo
    print(paste('p-value:', p_value))
    

    # make histogram plot
    df <- data.frame(difference = abs(permutation_diffs))
    df$category <- ifelse(df$difference > abs(final_obs), 'extreme', 'not')

    plot <- ggplot(df, aes(x = difference, fill = category)) +
        geom_histogram(bins = 100,
                        color = "black",
                        alpha = 0.7) +
        geom_vline(xintercept = abs(final_obs),
                    color = "red",
                    linetype = "dotted",
                    linewidth = 1
                    ) +
        labs(title = paste(comp,metric,"Permutation Test: Paired Sign Test Stat Differences"),
                x = paste("Sum of ",metric,"Differences"),
                y = "Frequency"
                ) +
        theme_classic() +
        scale_shape_manual(values = SHAPE,) +
        scale_colour_manual(values = c('green', 'black')) +
        scale_fill_manual(values = c('green', 'black'))

  print(plot)

  } else {
    stop("Invalid alternative (less, greater, or two-sided.")
  }
}

print('f1 Macro Results: Simple')
sink("/Users/gabrielketron/class_kw/simple_tpot_f1_results.txt")
permutation_test(simplefirstandsimpleresults$Exp3impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="Simple ", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_tpot_f1.png")

print('f1 Macro Results: Simple First')
sink("/Users/gabrielketron/class_kw/simple_first_f1_results.txt")
permutation_test(simplefirstandsimpleresults$Exp2impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="Simple First ", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_first_f1.png")

print('f1 Macro Results: Mixed')
sink("/Users/gabrielketron/class_kw/complex_f1_results.txt")
permutation_test(expertandcomplexresults$Exp3impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="Mixed", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/complex_f1.png")

print('f1 Macro Results: Impute First')
sink("/Users/gabrielketron/class_kw/impute_first_f1_results.txt")
permutation_test(spinefirstresults$Exp2impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="Impute First ", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/impute_first_f1.png")

print('f1 Macro Results: Complex Only')
sink("/Users/gabrielketron/class_kw/nonsimple_f1_new_results.txt")
permutation_test(nonsimple$Exp3impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="Complex Only ", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/nonsimple_f1_new.png")

print('f1 Macro Results: No Imputation')
sink("/Users/gabrielketron/class_kw/noimp_f1_results.txt")
permutation_test(nonsimple$Exp2impute_f1, expertandcomplexresults$Exp2impute_f1, seed = 1, alternative = "t", comp="No Imputation ", metric="F1 Macro ")
sink()
ggsave("/Users/gabrielketron/class_kw/noimp_f1.png")

print('Log Loss Results: Simple')
sink("/Users/gabrielketron/class_kw/simple_tpot_logloss_results.txt")
permutation_test(simplefirstandsimpleresults$Exp3impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="Simple ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_tpot_logloss.png")

print('Log Loss Results: Simple First')
sink("/Users/gabrielketron/class_kw/simple_first_logloss_results.txt")
permutation_test(simplefirstandsimpleresults$Exp2impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="Simple First ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_first_logloss.png")

print('Log Loss Results: Mixed')
sink("/Users/gabrielketron/class_kw/complex_logloss_results.txt")
permutation_test(expertandcomplexresults$Exp3impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="Mixed ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/complex_logloss.png")

print('Log Loss Results: Impute First')
sink("/Users/gabrielketron/class_kw/impute_first_logloss_results.txt")
permutation_test(spinefirstresults$Exp2impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="Impute First ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/impute_first_logloss.png")

print('Log Loss Results: Complex Only')
sink("/Users/gabrielketron/class_kw/nonsimple_logloss_results.txt")
permutation_test(nonsimple$Exp3impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="Complex Only ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/nonsimple_logloss.png")

print('Log Loss Results: No Imputation')
sink("/Users/gabrielketron/class_kw/noimp_logloss_results.txt")
permutation_test(nonsimple$Exp2impute_logloss, expertandcomplexresults$Exp2impute_logloss, seed = 2, alternative = "t", comp="No Imputation ", metric="Log Loss ")
sink()
ggsave("/Users/gabrielketron/class_kw/noimp_logloss.png")


#print('Log Loss Results:')
#permutation_test(results$Exp2impute_logloss, oldresults$Exp2impute_logloss, seed = 2, alternative = "t")

#ggsave("/Users/gabrielketron/class_kw/spine_logloss.png")
#print('Balanced Accuracy Results:')
#permutation_test(results$Exp2impute_balanced_accuracy, oldresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t")

print('Balanced Accuracy Results: Simple')
sink("/Users/gabrielketron/class_kw/simple_tpot_balanced_accuracy_results.txt")
permutation_test(simplefirstandsimpleresults$Exp3impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="Simple ", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_tpot_balanced_accuracy.png")

print('Balanced Accuracy Results: Simple First')
sink("/Users/gabrielketron/class_kw/simple_first_balanced_accuracy_results.txt")
permutation_test(simplefirstandsimpleresults$Exp2impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="Simple First ", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_first_balanced_accuracy.png")

print('Balanced Accuracy Results: Mixed')
sink("/Users/gabrielketron/class_kw/complex_balanced_accuracy_results.txt")
permutation_test(expertandcomplexresults$Exp3impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="Mixed ", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/complex_balanced_accuracy.png")

print('Balanced Accuracy Results: Impute First')
sink("/Users/gabrielketron/class_kw/impute_first_balanced_accuracy_results.txt")
permutation_test(spinefirstresults$Exp2impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="Impute First ", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/impute_first_balanced_accuracy.png")

print('Balanced Accuracy Results: Complex Only ')
sink("/Users/gabrielketron/class_kw/nonsimple_balanced_accuracy_results.txt")
permutation_test(nonsimple$Exp3impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="Complex Only ", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/nonsimple_balanced_accuracy.png")

print('Balanced Accuracy Results: No Imputation')
sink("/Users/gabrielketron/class_kw/noimp_balanced_accuracy_results.txt")
permutation_test(nonsimple$Exp2impute_balanced_accuracy, expertandcomplexresults$Exp2impute_balanced_accuracy, seed = 3, alternative = "t", comp="No Imputation", metric="Balanced Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/noimp_balanced_accuracy.png")

#ggsave("/Users/gabrielketron/class_kw/spine_first_balanced_accuracy.png")
#print('Accuracy Results:')
#permutation_test(results$Exp2impute_accuracy, oldresults$Exp2impute_accuracy, seed = 4, alternative = "t")

print('Accuracy Results: Simple')
sink("/Users/gabrielketron/class_kw/simple_tpot_accuracy_results.txt")
permutation_test(simplefirstandsimpleresults$Exp3impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="Simple ", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_tpot_accuracy.png")

print('Accuracy Results: Simple First')
sink("/Users/gabrielketron/class_kw/simple_first_accuracy_results.txt")
permutation_test(simplefirstandsimpleresults$Exp2impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="Simple First ", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_first_accuracy.png")

print('Accuracy Results: Mixed')
sink("/Users/gabrielketron/class_kw/complex_accuracy_results.txt")
permutation_test(expertandcomplexresults$Exp3impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="Mixed ", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/complex_accuracy.png")

print('Accuracy Results: Impute First')
sink("/Users/gabrielketron/class_kw/impute_first_accuracy_results.txt")
permutation_test(spinefirstresults$Exp2impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="Impute First ", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/impute_first_accuracy.png")

print('Accuracy Results: Complex Only ')
sink("/Users/gabrielketron/class_kw/nonsimple_accuracy_results.txt")
permutation_test(nonsimple$Exp3impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="Complex Only ", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/nonsimple_accuracy.png")

print('Accuracy Results: No Imputation')
sink("/Users/gabrielketron/class_kw/noimp_accuracy_results.txt")
permutation_test(nonsimple$Exp2impute_accuracy, expertandcomplexresults$Exp2impute_accuracy, seed = 4, alternative = "t", comp="No Imputation", metric="Accuracy ")
sink()
ggsave("/Users/gabrielketron/class_kw/noimp_accuracy.png")


#ggsave("/Users/gabrielketron/class_kw/spine_first_accuracy.png")
#print('AUROC Results:')
#permutation_test(results$Exp2impute_auroc, oldresults$Exp3impute_auroc, seed = 5, alternative = "t")

print('AUROC Results: Simple')
sink("/Users/gabrielketron/class_kw/simple_tpot_auroc_results.txt")
permutation_test(simplefirstandsimpleresults$Exp3impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="Simple ", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_tpot_auroc.png")

print('AUROC Results: Simple First')
sink("/Users/gabrielketron/class_kw/simple_first_auroc_results.txt")
permutation_test(simplefirstandsimpleresults$Exp2impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="Simple First ", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/simple_first_auroc.png")

print('AUROC Results: Mixed')
sink("/Users/gabrielketron/class_kw/complex_auroc_results.txt")
permutation_test(expertandcomplexresults$Exp3impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="Mixed ", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/complex_auroc.png")

print('AUROC Results: Impute First')
sink("/Users/gabrielketron/class_kw/impute_first_auroc_results.txt")
permutation_test(spinefirstresults$Exp2impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="Impute First ", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/impute_first_auroc.png")

print('AUROC Results: Complex Only')
sink("/Users/gabrielketron/class_kw/nonsimple_auroc_results.txt")
permutation_test(nonsimple$Exp3impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="Complex Only ", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/nonsimple_auroc.png")

print('Accuracy Results: No Imputation')
sink("/Users/gabrielketron/class_kw/noimp_auroc_results.txt")
permutation_test(nonsimple$Exp2impute_auroc, expertandcomplexresults$Exp2impute_auroc, seed = 5, alternative = "t", comp="No Imputation", metric="AUROC ")
sink()
ggsave("/Users/gabrielketron/class_kw/noimp_auroc.png")

#ggsave("/Users/gabrielketron/class_kw/headtohead_auroc.png")