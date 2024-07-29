import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import time
import sklearn.datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer



def main():
    # Read in arguements
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')

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
    classification_id_list = []
    regression_id_list = []
    print('starting loops')
    start = time.time()
    utils.loop_through_tasks(class_experiments, classification_id_list, base_save_folder, num_runs, 'c')
    utils.loop_through_tasks(reg_experiments, regression_id_list, base_save_folder, num_runs, 'r')
    stop = time.time()
    duration = stop - start
    print('full run takes')
    print(duration/3600)
    print('hours')


if __name__ == '__main__':
    main()
    print("DONE")