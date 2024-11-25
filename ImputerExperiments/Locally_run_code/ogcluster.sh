#!/bin/bash 
#$ -pe smp 16
#$ -N tpot2-impute
#$ -cwd
#$ -q all.q
#$ -t 1-72
#$ -j y
#$ -o ../data/logs/outputs/oldhpcoutput.$JOB_ID_$TASK_ID.out


RUN=${SGE_TASK_ID:-1}

echo "Run: ${RUN}"

module load git
module unload python3/3.12.7
module load python3/3.9.16

python3.9 -m venv /common/ketrong/tpotexp/env3

source /common/ketrong/tpotexp/env3/bin/activate
echo $VIRTUAL_ENV

pip install -e /common/ketrong/tpotexp/tpot2
pip install -r /common/ketrong/tpotexp/tpot2/ImputerExperiments/requirements_.txt

echo RunStart

which pip
which python3.9

python3.9 /common/ketrong/tpotexp/tpot2/ImputerExperiments/Locally_run_code/mainog.py \
--n_jobs 16 \
--savepath ../data \
--num_runs ${RUN} \