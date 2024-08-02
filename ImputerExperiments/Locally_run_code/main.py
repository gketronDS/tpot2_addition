import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import time
import sklearn.datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import utils



def main():
    # Read in arguements
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')

    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)
    total_duration = 360000

    classifier_pipeline_full = tpot2.search_spaces.pipelines.SequentialPipeline([
        tpot2.config.get_search_space("imputers"), 
        tpot2.config.get_search_space("classifiers"),
    ])
    classifier_pipeline_simple = tpot2.search_spaces.pipelines.SequentialPipeline([
        tpot2.config.get_search_space(["SimpleImputer"]), 
        tpot2.config.get_search_space("classifiers"),
    ])

    regression_pipeline_full = tpot2.search_spaces.pipelines.SequentialPipeline([
        tpot2.config.get_search_space("imputers"), 
        tpot2.config.get_search_space("regressors"),
    ])
    regression_pipeline_simple = tpot2.search_spaces.pipelines.SequentialPipeline([
        tpot2.config.get_search_space(["SimpleImputer"]), 
        tpot2.config.get_search_space("regressors"),
    ])

    classification_full = {
                    'scorers':['f1_weighted'],
                    'scorers_weights':[1],
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
                    'n_jobs':n_jobs,
                    'cv': 10,
                    'verbose':5, 
                    'max_time_seconds': total_duration,
                    'max_eval_time_seconds':60*10, 
                    'classification' : True,
                    'search_space': classifier_pipeline_full,
                    'preprocessing':False,
    }
    classification_simple = {
                    'scorers':['f1_weighted'],
                    'scorers_weights':[1],
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
                    'n_jobs':n_jobs,
                    'cv': 10,
                    'verbose':5, 
                    'max_time_seconds': total_duration,
                    'max_eval_time_seconds':60*10, 
                    'classification' : True,
                    'search_space': classifier_pipeline_simple,
                    'preprocessing':False,
    }

    regression_full = {
                    'scorers':['neg_root_mean_square_error'],
                    'scorers_weights':[1],
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
                    'n_jobs':n_jobs,
                    'cv': 10,
                    'verbose':5, 
                    'max_time_seconds': total_duration,
                    'max_eval_time_seconds':60*10, 
                    'classification' : False,
                    'search_space': regression_pipeline_full,
                    'preprocessing':False,
    }
    regression_simple = {
                    'scorers':['neg_root_mean_square_error'],
                    'scorers_weights':[1],
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
                    'n_jobs':n_jobs,
                    'cv': 10,
                    'verbose':5, 
                    'max_time_seconds': total_duration,
                    'max_eval_time_seconds':60*10, 
                    'classification' : False,
                    'search_space': regression_pipeline_simple,
                    'preprocessing':False,
    }

    class_experiments = [
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'class_full',
            'params': classification_full,
            },
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'class_simple',
            'params': classification_simple,
            },
            ]
    reg_experiments = [
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'reg_full',
            'params': regression_full,
            },
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'reg_simple',
            'params': regression_simple,
            },
            ]
    #try with 67 / 69 benchmark sets
    '''
    classification_id_list = [6, 26, 30, 32, 137, 151, 183, 184, 251, 310, 375, 725,
                              728, 737, 803, 847, 871, 881, 901, 923, 1046, 
                              1120, 1220, 1558, 1526, 1507, 1489, 1496, 1481,
                              1471, 4552, 1459, 4135, 40498, 40497, 40677, 
                              40685, 23395, 40983, 41027, 23517, 40701, 40922,
                              41671, 41146, 42192]
    regression_id_list = [189, 197, 198, 215, 216, 218, 1193, 1199, 42183, 
                          42545, 42225, 42712, 287, 42688, 23515]
    '''
    classification_id_list = [6]
    regression_id_list = [189]

    
    print('starting loops')
    start = time.time()
    utils.loop_through_tasks(class_experiments, classification_id_list, 
                             base_save_folder, num_runs, 'c', n_jobs=n_jobs)
    utils.loop_through_tasks(reg_experiments, regression_id_list, 
                             base_save_folder, num_runs, 'r', n_jobs=n_jobs)
    stop = time.time()
    duration = stop - start
    print('full run takes')
    print(duration/3600)
    print('hours')


if __name__ == '__main__':
    main()
    print("DONE")