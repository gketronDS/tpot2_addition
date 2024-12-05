#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 110:00:00
#SBATCH --mem=100000
#SBATCH --job-name=tpot2-class-R
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=Gabriel.Ketron@cshs.org
#SBATCH --mail-user=gketron@uci.edu
#SBATCH -o ../data/logs/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=1

RUN=${SLURM_ARRAY_TASK_ID:-1}

echo "Run: ${RUN}"

module purge
module load R/4.4.0

R
install.packages(c("tidyverse","agridat", "ggplot2", "ghibli", "ggdist", "ggstatsplot", "ISLR"))
yes
70
q()

Rscript class_analysis.R
