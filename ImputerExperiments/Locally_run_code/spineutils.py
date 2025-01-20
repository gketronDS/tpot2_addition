import sklearn.preprocessing
import openml
import tpot2
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             f1_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
import traceback
import dill as pickle
import os
import time
import openml
import sklearn.datasets
import numpy as np
import time
import random
import sklearn.model_selection
import torch
from scipy import optimize
import pandas as pd
import autoimpute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from datetime import datetime
import random

import tpot2.tpot_estimator


def score(est, X, y, r_or_c):
    if r_or_c == 'c':
        try:
            this_auroc_score = sklearn.metrics.get_scorer("roc_auc_ovr")(est, X, y)
        except:
            y_preds = est.predict(X)
            y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
            this_explained_score = sklearn.metrics.explained_variance_score(y, y_preds_onehot)
        
        try:
            this_logloss = sklearn.metrics.get_scorer("neg_log_loss")(est, X, y)*-1
        except:
            y_preds = est.predict(X)
            y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
            this_logloss = log_loss(y, y_preds_onehot)

        this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
        this_balanced_accuracy_score = sklearn.metrics.get_scorer("balanced_accuracy")(est, X, y)
        this_f1_score = sklearn.metrics.get_scorer("f1_macro")(est, X, y)

        return { "auroc": this_auroc_score,
                "accuracy": this_accuracy_score,
                "balanced_accuracy": this_balanced_accuracy_score,
                "logloss": this_logloss,
                "f1": this_f1_score,
                }
    else:
        try: 
            this_explained_score = sklearn.metrics.get_scorer("explained_variance")(est, X, y)
        except:
            y_preds = est.predict(X)
            this_explained_score = sklearn.metrics.explained_variance_score(y, y_preds)
        try: 
            this_rmse = sklearn.metrics.get_scorer('neg_root_mean_squared_error')(est, X, y)*-1
        except:
            y_preds = est.predict(X)
            this_rmse = sklearn.metrics.root_mean_squared_error(y, y_preds)*-1

        this_r2_score = sklearn.metrics.get_scorer("r2")(est, X, y)
        return { "explained_var": this_explained_score,
                "r2": this_r2_score,
                "rmse": this_rmse,
    }


#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(base_save_folder, task_id, r_or_c):
    cached_data_path = f"{base_save_folder}/{r_or_c}/{task_id}/{task_id}.pkl"
    print(cached_data_path)
    #check = False
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train_i, X_train_u, y_train, X_test_i, X_test_u, y_test = d['X_train_i'], d['X_train_u'], d['y_train'], d['X_test_i'], d['X_test_u'], d['y_test']
    #kwargs = {'force_refresh_cache': True}
    else:
        preimpute = pd.read_csv(base_save_folder+f'/spine_data/combined_cohort_imputed_{task_id}.csv')
        nonimpute = pd.read_csv(base_save_folder+f'/spine_data/combined_cohort_unimputed_{task_id}.csv')
        preimpute['SURGERY_DATE_TIME'] = preimpute['SURGERY_DATE_TIME'].apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))
        nonimpute['SURGERY_DATE_TIME'] = nonimpute['SURGERY_DATE_TIME'].apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))

        preimpute['SURGERY_DATE_TIME'] = preimpute['SURGERY_DATE_TIME'].apply(lambda x: x.timestamp())
        nonimpute['SURGERY_DATE_TIME'] = nonimpute['SURGERY_DATE_TIME'].apply(lambda x: x.timestamp())
        preimpute.drop('PAT_ID', axis=1, inplace=True)
        nonimpute.drop('PAT_ID', axis=1, inplace=True)

        y = preimpute.iloc[:, -1:]
        X_i = preimpute.iloc[:, :-1]
        X_u = nonimpute.iloc[:, :-1]

        def Replace(i):
            try:
                float(i)
                return float(i)
            except:
                return np.nan

        X_i = X_i.applymap(func=Replace)
        X_u = X_u.applymap(func=Replace)
        print(X_i.isna().sum().sum())
        print(X_u.isna().sum().sum())
        X_i = X_i.apply(pd.to_numeric, errors='coerce')
        X_u = X_u.apply(pd.to_numeric, errors='coerce')
        print(X_i.isna().sum().sum())
        print(X_u.isna().sum().sum())
        #X_i.to_csv(f'{base_save_folder}/{r_or_c}/{task_id}/{task_id}_i.csv')
        #X_u.to_csv(f'{base_save_folder}/{r_or_c}/{task_id}/{task_id}_u.csv')

        print(y)
        print(X_i)
        #pd.set_option('display.max_columns', None)
        #print(X)
        #print(type(X))
        if r_or_c =='c':
            X_train_i, X_test_i, y_train, y_test = train_test_split(X_i, y, test_size=0.2, stratify=y)
        else: 
            X_train_i, X_test_i, y_train, y_test = train_test_split(X_i, y, test_size=0.2)

        index_train = X_train_i.index
        index_test = X_test_i.index
        X_train_u = X_u.loc[index_train]
        X_test_u = X_u.loc[index_test]
        '''
        X_train_u.reset_index(drop=True, inplace=True)
        X_train_i.reset_index(drop=True, inplace=True) 
        y_train.reset_index(drop=True, inplace=True)
        X_test_u.reset_index(drop=True, inplace=True)
        X_test_i.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        '''
        print(len(X_train_i))
        print(len(X_train_u))
        print(len(y_train))
        print('---------')
        print(len(X_test_i))
        print(len(X_test_u))
        print(len(y_test))


        #print(X_train_u)
        print(type(X_train_u))
        

        if task_id == 42712:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.MinMaxScaler()
                )
        else:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline( 
                sklearn.preprocessing.MinMaxScaler()
                )
        
        
        X_train_i = preprocessing_pipeline.fit_transform(X_train_i)
        X_test_i = preprocessing_pipeline.transform(X_test_i)
        X_train_u = preprocessing_pipeline.fit_transform(X_train_u)
        X_test_u = preprocessing_pipeline.transform(X_test_u)

        print(pd.DataFrame(X_train_i))
        print("///////")
        print(pd.DataFrame(X_test_i))
        print("///////")
        print(pd.DataFrame(X_train_u))
        print("///////")
        print(pd.DataFrame(X_test_u))
        print("///////")
        #print(pd.DataFrame(X_train_u))
        #print(type(X_train))
        #print(pd.DataFrame(X_test_u))
        #print(type(X_test))
        
        if r_or_c =='c':
            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        
        print(type(X_train_i))
        print(y_train)
        print("///////")
        print(y_test)
        

        d = {"X_train_i": X_train_i, "X_train_u": X_train_u, "y_train": y_train, "X_test_u": X_test_u, "X_test_i": X_test_i, "y_test": y_test}
        if not os.path.exists(f"{base_save_folder}"):
            os.makedirs(f"{base_save_folder}")
        with open(cached_data_path, "wb") as f:
            pickle.dump(d, f)

    return X_train_u, X_train_i, y_train, X_test_u, X_test_i, y_test


def loop_through_tasks(experiments, base_save_folder, num_runs, r_or_c, n_jobs):
    device = ("cpu")
    torch.set_default_device(device)
    match num_runs: 
        case 1: 
            outcome = 'home'
            num_iter = 1
            exp = experiments[0]
        case 2: 
            outcome = 'hosp'
            num_iter = 1
            exp = experiments[0]
        case 3: 
            outcome = 'los'
            num_iter = 1
            exp = experiments[0]
        case 4: 
            outcome = 'rehab'
            num_iter = 1
            exp = experiments[0]
        case 5: 
            outcome = 'SNF'
            num_iter = 1
            exp = experiments[0]
        case 6: 
            outcome = 'home'
            num_iter = 2
            exp = experiments[0]
        case 7: 
            outcome = 'hosp'
            num_iter = 2
            exp = experiments[0]
        case 8: 
            outcome = 'los'
            num_iter = 2
            exp = experiments[0]
        case 9: 
            outcome = 'rehab'
            num_iter = 2
            exp = experiments[0]
        case 10: 
            outcome = 'SNF'
            num_iter = 2
            exp = experiments[0]
        case 11: 
            outcome = 'home'
            num_iter = 3
            exp = experiments[0]
        case 12: 
            outcome = 'hosp'
            num_iter = 3
            exp = experiments[0]
        case 13: 
            outcome = 'los'
            num_iter = 3
            exp = experiments[0]
        case 14: 
            outcome = 'rehab'
            num_iter = 3
            exp = experiments[0]
        case 15: 
            outcome = 'SNF'
            num_iter = 3
            exp = experiments[0]
    save_folder = f"{base_save_folder}/{r_or_c}/{outcome}"
    time.sleep(random.random()*5)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    X_train_u, X_train_i, y_train, X_test_u, X_test_i, y_test = load_task(base_save_folder=base_save_folder, task_id=outcome, r_or_c= r_or_c)
    save_folder = f"{base_save_folder}/{r_or_c}/{outcome}/{exp['exp_name']}_{num_iter}"
    checkpoint_folder = f"{base_save_folder}/checkpoint/{r_or_c}/{outcome}/{exp['exp_name']}_{num_iter}"
    #print('loc5')
    time.sleep(random.random()*5)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    #print('loc6')
    time.sleep(random.random()*5)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    print("working on ")
    print(save_folder)

    start = time.time()
    time.sleep(random.random()*5)
    duration = time.time() - start
    print(duration)

    try:
        print("running experiment 1/3 - Does large hyperparameter space improve reconstruction accuracy over simple")
        #Simple Impute 
        missingnessmodels = ['MAR', 'MNAR', 'MCAR']
        random.seed(num_runs)
        ourmodel = random.choice(missingnessmodels)
        first_scores = {}
        X_train_u = pd.DataFrame(X_train_u)
        X_test_u = pd.DataFrame(X_test_u)
        
        if exp['exp_name'] == 'class_simple' or exp['exp_name'] == 'reg_simple':
            SimpleImputeSpace = autoimpute.AutoImputer(
                model_names=['SimpleImputer'],
                n_jobs=n_jobs, show_progress=False, random_state=num_iter, missing_type=ourmodel)
            impute_train = SimpleImputeSpace.fit_transform(X_train_u)
            print('simple fit')
            impute_test = SimpleImputeSpace.transform(X_test_u)
            print('simple transform')
            print(impute_train.isna().sum())
            print(impute_test.isna().sum())
            simple_rmse = SimpleImputeSpace.study.best_trial.value
            simple_space = SimpleImputeSpace.study.best_trial.params
            impute_train = impute_train.to_numpy()
            impute_test = impute_test.to_numpy()
            print(simple_rmse)
            print(simple_space)
            first_scores['impute_rmse'] = simple_rmse
            first_scores['impute_space'] = simple_space
        else:
            #Auto Impute 
            AutoImputeSpace = autoimpute.AutoImputer(model_names=['SimpleImputer', 'IterativeImputer', 'KNNImputer', 'GAIN', 'VAE'], n_jobs=n_jobs, show_progress=False, random_state=num_iter, missing_type=ourmodel)
            impute_train = AutoImputeSpace.fit_transform(X_train_u)
            print('auto fit')
            impute_test = AutoImputeSpace.transform(X_test_u)
            print('auto transform')
            print(impute_train.isna().sum())
            print(impute_test.isna().sum())
            auto_rmse = AutoImputeSpace.study.best_trial.value
            auto_space = AutoImputeSpace.study.best_trial.params
            impute_train = impute_train.to_numpy()
            impute_test = impute_test.to_numpy()
            print(auto_rmse)
            print(auto_space)
            first_scores['impute_rmse'] = auto_rmse
            first_scores['impute_space'] = auto_space

        print("running experiment 2/3 - Does reconstruction give good automl predictions")
        #this section trains off of original train data, and then tests on the original, the simpleimputed,
        #  and the autoimpute test data. This section uses the normal params since it is checking just for predictive preformance, 
        # not the role of various imputers in the tpot optimization space. 
        try: 
            os.remove(f"{checkpoint_folder}/population.pkl")
        except:
            print('no checkpoint to remove')

        exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
        if r_or_c == 'c':
            estimator_params = exp['params'].copy()
            estimator_params['search_space'] =  tpot2.search_spaces.pipelines.SequentialPipeline(
                [tpot2.config.get_search_space("classifiers")]
                )
            est = tpot2.TPOTEstimator(**estimator_params)
        else: 
            estimator_params = exp['params']
            estimator_params['search_space'] =  tpot2.search_spaces.pipelines.SequentialPipeline(
                [tpot2.config.get_search_space("regressors")]
                )
            est = tpot2.TPOTEstimator(**estimator_params)
        
        estimator_params = exp['params'].copy()
        estimator_params['search_space'] =  tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space("classifiers", random_state=num_iter)
            ])
        est = tpot2.TPOTEstimator(**estimator_params)
        print(est)
        print(est.evaluated_individuals)

        print('Start est fit')
        #print(np.count_nonzero(np.isnan(X_train_u)))
        #print(np.count_nonzero(np.isnan(y_train)))
        start = time.time()
        try:
            est.fit(impute_train, y_train)
        except:
            print('Failed')
            print(est.evalauted_individuals)
        '''
        try:
            model = sklearn.naive_bayes.GaussianNB()
            model.fit(X_train_i, y_train)
            print(model.score(X_test_i, y_test))
            #est.fit(impute_train, y_train)
        except:
            try:
                print(est.population)
            except Exception as e:
                trace =  traceback.format_exc() 
                pipeline_failure_dict = {"taskid": outcome, "exp_name": exp['exp_name'], "run": num_iter, "error": str(e), "trace": trace}
                print("failed on ")
                print(save_folder)
                print(e)
                print(trace)
        '''
        stop = time.time()
        duration = stop - start
        print('Fitted')
        if exp['automl'] is tpot2.TPOTClassifier:
            est.classes_ = est.fitted_pipeline_.classes_
        print(est.fitted_pipeline_)
        print('score start')
        train_score = score(est, impute_train, y_train, r_or_c=r_or_c)
        print('train score:', train_score)
        ori_test_score = score(est, X_test_i, y_test, r_or_c=r_or_c)
        print('original test score:', ori_test_score)
        start2 = time.time()
        imputed_test_score = score(est, impute_test, y_test, r_or_c=r_or_c)
        stop2 = time.time()
        inferenceduration2 = stop2 - start2
        print('imputed test score:', imputed_test_score)
        print('score end')
        train_score = {f"train_{k}": v for k, v in train_score.items()}
        first_scores['train_score'] = train_score
        first_scores['ori_test_score']=ori_test_score
        first_scores['imputed_test_score'] = imputed_test_score
        first_scores["start"] = start
        first_scores["taskid"] = outcome
        first_scores["exp_name"] = 'Imputed_Predictive_Capacity'
        first_scores["duration"] = duration
        first_scores["inference_time"] = inferenceduration2
        first_scores["run"] = num_iter
        first_scores["fit_model"] = est.fitted_pipeline_
        first_scores["r_or_c"] = r_or_c

        if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
            with open(f"{save_folder}/first_evaluated_individuals.pkl", "wb") as f:
                pickle.dump(est.evaluated_individuals, f)
                print('estimator working as intended')
        print('check intended')
        with open(f"{save_folder}/first_fitted_pipeline.pkl", "wb") as f:
            pickle.dump(est.fitted_pipeline_, f)

        with open(f"{save_folder}/first_scores.pkl", "wb") as f:
            pickle.dump(first_scores, f)

        print('pre-imputed Finished')
        
        '''
        print("running experiment 3/3 - What is the best automl settings?")
        try: 
            os.remove(f"{checkpoint_folder}/population.pkl")
        except:
            print('no checkpoint to remove')
        
        exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
        
        tpot_space = exp['automl']
        print(exp['params']['search_space'])
        print('Start tpot fit')
        start = time.time()
        tpot_space.fit(X_train_u, y_train)
        stop = time.time()
        duration = stop - start
        print('Fitted')
        if exp['automl'] is tpot2.TPOTClassifier:
            tpot_space.classes_ = tpot_space.fitted_pipeline_.classes_
        print(tpot_space.fitted_pipeline_)
        X_train_transform = tpot_space.fitted_pipeline_[0].transform(X_train_u)
        print('transform worked')
        #rmse_loss_train3 = autoimpute.rmse_loss(ori_data=X_train_i, imputed_data=X_train_transform, data_m=np.multiply(mask_train.to_numpy(),1))
        print('try transform')
        X_test_transform = tpot_space.fitted_pipeline_[0].transform(X_test_u)
        print('transform worked')
        #rmse_loss_test3 = autoimpute.rmse_loss(ori_data=X_test, imputed_data=X_test_transform, data_m=np.multiply(mask_test.to_numpy(),1))
        print('score start')
        train_score = score(tpot_space, X_train_transform, y_train, r_or_c=r_or_c)
        print('train score:', train_score)
        start = time.time()
        test_score = score(tpot_space, X_test_transform, y_test, r_or_c=r_or_c)
        stop = time.time()
        duration2 = stop - start
        print('test score:', test_score)
        ori_test_score = score(tpot_space, X_test_i, y_test, r_or_c=r_or_c)
        print('original test score:', ori_test_score)
        print('score end')
        tpot_space_scores = {}
        train_score = {f"train_{k}": v for k, v in train_score.items()}
        
        tpot_space_scores['train_score'] = train_score
        tpot_space_scores['test_score']=test_score    
        tpot_space_scores['ori_test_score']=ori_test_score    
        tpot_space_scores["start"] = start
        tpot_space_scores["taskid"] = outcome
        tpot_space_scores["exp_name"] = exp['exp_name']
        tpot_space_scores["duration"] = duration
        tpot_space_scores["inference_time"] = duration2
        tpot_space_scores["run"] = num_iter
        tpot_space_scores["fit_model"] = tpot_space.fitted_pipeline_
        tpot_space_scores["r_or_c"] = r_or_c


        if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.tpot_estimator.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
            with open(f"{save_folder}/tpot_space_evaluated_individuals.pkl", "wb") as f:
                pickle.dump(tpot_space.evaluated_individuals, f)

        with open(f"{save_folder}/tpot_space_fitted_pipeline.pkl", "wb") as f:
            pickle.dump(tpot_space.fitted_pipeline_, f)

        with open(f"{save_folder}/tpot_space_scores.pkl", "wb") as f:
            pickle.dump(tpot_space_scores, f)
        '''
        #return
        
    except Exception as e:
        trace =  traceback.format_exc() 
        pipeline_failure_dict = {"taskid": outcome, "exp_name": exp['exp_name'], "run": num_iter, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

        with open(f"{save_folder}/failed.pkl", "wb") as f:
            pickle.dump(pipeline_failure_dict, f)
        return 
    
    print('num_run')
    print(num_iter)
    print(exp['exp_name'])
    print('finished')
    print('all finished')
    return


### Additional Stuff GKetron Added
def add_missing(X, add_missing = 0.05, missing_type = 'MAR'):
    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X)
    missing_mask = X
    missing_mask = missing_mask.mask(missing_mask.isna(), True)
    missing_mask = missing_mask.mask(missing_mask.notna(), False)
    X = X.mask(X.isna(), 0)
    T = torch.tensor(X.to_numpy())

    match missing_type:
        case 'MAR':
            out = MAR(T, [add_missing])
        case 'MCAR':
            out = MCAR(T, [add_missing])
        case 'MNAR':
            out = MNAR_mask_logistic(T, [add_missing])
    
    masked_set = pd.DataFrame(out['Mask'].numpy())
    missing_combo = (missing_mask | masked_set.isna())
    masked_set = masked_set.mask(missing_combo, True)
    masked_set.columns = X.columns.values
    #masked_set = masked_set.to_numpy()

    missing_set = pd.DataFrame(out['Missing'].numpy())
    missing_set.columns = X.columns.values
    #missing_set = missing_set.to_numpy()

    return missing_set, masked_set

"""BEYOND THIS POINT WRITTEN BY Aude Sportisse, Marine Le Morvan and Boris Muzellec - https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values"""

def MCAR(X, p_miss):
    out = {'X': X.double()}
    for p in p_miss: 
        mask = (torch.rand(X.shape) < p).double()
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out

def MAR(X,p_miss,p_obs=0.5):
    out = {'X': X.double()}
    for p in p_miss:
        n, d = X.shape
        mask = torch.zeros(n, d).bool()
        num_no_missing = max(int(p_obs * d), 1)
        num_missing = d - num_no_missing
        obs_samples = np.random.choice(d, num_no_missing, replace=False)
        copy_samples = np.array([i for i in range(d) if i not in obs_samples])
        len_obs = len(obs_samples)
        len_na = len(copy_samples)
        coeffs = torch.randn(len_obs, len_na).double()
        Wx = X[:, obs_samples].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
        coeffs.double()
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, obs_samples].mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, obs_samples].mm(coeffs) + intercepts)
        ber = torch.rand(n, len_na)
        mask[:, copy_samples] = ber < ps
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out

def MNAR_mask_logistic(X, p_miss, p_params =.5, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    out = {'X_init_MNAR': X.double()}
    for p in p_miss: 
        n, d = X.shape
        to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = torch.from_numpy(X)
        mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
        d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
        d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model
        ### Sample variables that will be parameters for the logistic regression:
        idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)
        ### Other variables will have NA proportions selected by a logistic model
        ### The parameters of this logistic model are random.
        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        len_obs = len(idxs_params)
        len_na = len(idxs_nas)
        coeffs = torch.randn(len_obs, len_na).double()
        Wx = X[:, idxs_params].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
        coeffs.double()
        ### Pick the intercepts to have a desired amount of missing values
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, idxs_params].mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)
        ber = torch.rand(n, d_na)
        mask[:, idxs_nas] = ber < ps
        ## If the inputs of the logistic model are excluded from MNAR missingness,
        ## mask some values used in the logistic model at random.
        ## This makes the missingness of other variables potentially dependent on masked values
        if exclude_inputs:
            mask[:, idxs_params] = torch.rand(n, d_params) < p
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out
