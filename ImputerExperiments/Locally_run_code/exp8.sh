#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 120:00:00
#SBATCH --mem=330000
#SBATCH --job-name=tpot2-impute
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=Gabriel.Ketron@cshs.org
#SBATCH --mail-user=gketron@uci.edu
#SBATCH -o ../data/logs/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=1-72%12


RUN=${SLURM_ARRAY_TASK_ID:-1}

echo "Run: ${RUN}"

module load git/2.33.1

source /common/ketrong/tpotexp/env/bin/activate
'''
pip install -e tpot2
pip install -r tpot2/ImputerExperiments/requirements_.txt
'''

echo RunStart

srun -u python3.10 main8.py \
--n_jobs 16 \
--savepath ../data \
--num_runs ${RUN} \