
echo "Initalize"


source ../../../env2/bin/activate

echo RunStart

python main.py \
--n_jobs 8 \
--savepath ../data \
--num_runs 3