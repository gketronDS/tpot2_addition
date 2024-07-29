
echo "Initalize"


source ../../../env2/bin/activate

echo RunStart

python main.py \
--n_jobs 8 \
--savepath logs \
--num_runs 3