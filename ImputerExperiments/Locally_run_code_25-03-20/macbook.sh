echo "Run: 1"

module load git/2.33.1

source /Users/gabrielketron/tpot2_addimputers/env2/bin/activate
'''
pip install -e tpot2
pip install -r tpot2/ImputerExperiments/requirements_.txt
'''

echo RunStart

python3.10 main.py \
--n_jobs 4 \
--savepath ../data \
--num_runs 1 \