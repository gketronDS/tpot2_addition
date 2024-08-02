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


def score(est, X, y, r_or_c):
    if r_or_c == 'c':
        try:
            this_auroc_score = sklearn.metrics.get_scorer("roc_auc_ovr")(est, X, y)
        except:
            y_preds = est.predict(X)
            y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
            this_explained_score = sklearn.metrics.explained_variance_score(y, y_preds_onehot, multi_class="ovr")
        
        try:
            this_logloss = sklearn.metrics.get_scorer("neg_log_loss")(est, X, y)*-1
        except:
            y_preds = est.predict(X)
            y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
            this_logloss = log_loss(y, y_preds_onehot)

        this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
        this_balanced_accuracy_score = sklearn.metrics.get_scorer("balanced_accuracy")(est, X, y)
        this_f1_score = sklearn.metrics.get_scorer("f1_weighted")(est, X, y)

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
    
    cached_data_path = f"{base_save_folder}/{task_id}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        #kwargs = {'force_refresh_cache': True}
        task = openml.datasets.get_dataset(task_id)
        X, y, _, _  = task.get_data(dataset_format="dataframe")
        print(X)
        print(y)
        if y is None: 
            y = X.iloc[:, -1:]
            X = X.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        preprocessing_pipeline = sklearn.pipeline.make_pipeline(
            tpot2.builtin_modules.ColumnSimpleImputer(
                "categorical", strategy='most_frequent'), 
            tpot2.builtin_modules.ColumnSimpleImputer(
                "numeric", strategy='mean'), 
                tpot2.builtin_modules.ColumnOneHotEncoder(
                    "categorical", min_frequency=0.001, handle_unknown="ignore")
            )
        X_train = preprocessing_pipeline.fit_transform(X_train)
        X_test = preprocessing_pipeline.transform(X_test)

        X_train = sklearn.preprocessing.normalize(X_train)
        X_test = sklearn.preprocessing.normalize(X_test)

        if r_or_c =='c':
            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        if not os.path.exists(f"{base_save_folder}"):
            os.makedirs(f"{base_save_folder}")
        with open(cached_data_path, "wb") as f:
            pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


def loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs, r_or_c, n_jobs):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(True)
    for taskid in task_id_lists:
        save_folder = f"{base_save_folder}/{r_or_c}/{taskid}"
        time.sleep(random.random()*5)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        X_train, y_train, X_test, y_test = load_task(base_save_folder=save_folder, task_id=taskid, r_or_c= r_or_c)
        for level in [0.01, 0.1, 0.3, 0.5]:
                for type_1 in ['MAR', 'MCAR', 'MNAR']:
                    X_train = pd.DataFrame(X_train)
                    X_test = pd.DataFrame(X_test)
                    X_train_M, mask_train = add_missing(X_train, add_missing=level, missing_type=type_1)
                    X_test_M, mask_test = add_missing(X_test, add_missing=level, missing_type=type_1)
                    X_train_n = X_train_M.to_numpy()
                    X_test_n = X_test_M.to_numpy()
                    for exp in experiments:
                        for num_run in range(num_runs):
                            #print('loc4')
                            levelstr = str(level)
                            save_folder = f"{base_save_folder}/{r_or_c}/{taskid}/{exp['exp_name']}_{type_1}_{levelstr}_{num_run}"
                            checkpoint_folder = f"{base_save_folder}/checkpoint/{r_or_c}/{taskid}/{exp['exp_name']}_{type_1}_{levelstr}_{num_run}"
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
                                all_scores = {}
                                if exp['exp_name'] == 'class_simple' or exp['exp_name'] == 'reg_simple':
                                    SimpleImputeSpace = autoimpute.AutoImputer(
                                        missing_type=type_1, 
                                        model_names=['SimpleImputer'], 
                                        n_jobs=n_jobs, show_progress=False, random_state=num_runs)
                                    impute_train = SimpleImputeSpace.fit_transform(X_train_M)
                                    print('simple fit')
                                    impute_test = SimpleImputeSpace.transform(X_test_M)
                                    print('simple transform')
                                    print(impute_test)
                                    simple_rmse = SimpleImputeSpace.study.best_trial.value
                                    simple_space = SimpleImputeSpace.study.best_trial.params
                                    impute_train = impute_train.to_numpy()
                                    impute_test = impute_test.to_numpy()
                                    print(simple_rmse)
                                    print(simple_space)
                                    all_scores['impute_rmse'] = simple_rmse
                                    all_scores['impute_space'] = simple_space
                                else:
                                    #Auto Impute 
                                    AutoImputeSpace = autoimpute.AutoImputer(missing_type=type_1, model_names=['SimpleImputer', 'IterativeImputer', 'KNNImputer', 'GAIN'], n_jobs=n_jobs, show_progress=False, random_state=num_runs)
                                    impute_train = AutoImputeSpace.fit_transform(X_train_M)
                                    print('auto fit')
                                    impute_test = AutoImputeSpace.transform(X_test_M)
                                    print('auto transform')
                                    print(impute_test)
                                    auto_rmse = AutoImputeSpace.study.best_trial.value
                                    auto_space = AutoImputeSpace.study.best_trial.params
                                    impute_train = impute_train.to_numpy()
                                    impute_test = impute_test.to_numpy()
                                    print(auto_rmse)
                                    print(auto_space)
                                    all_scores['impute_rmse'] = auto_rmse
                                    all_scores['impute_space'] = auto_space
                                
                                print("running experiment 2/3 - Does reconstruction give good automl predictions")
                                #this section trains off of original train data, and then tests on the original, the simpleimputed,
                                #  and the autoimpute test data. This section uses the normal params since it is checking just for predictive preformance, 
                                # not the role of various imputers in the tpot optimization space. 

                                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                                if r_or_c == 'c':
                                    estimator_params = exp['params']
                                    estimator_params['search_space'] =  tpot2.search_spaces.pipelines.SequentialPipeline(
                                        [tpot2.config.get_search_space("classifiers"),]
                                        )
                                    est = exp['automl'](**estimator_params)
                                else: 
                                    estimator_params = exp['params']
                                    estimator_params['search_space'] =  tpot2.search_spaces.pipelines.SequentialPipeline(
                                        [tpot2.config.get_search_space("regressors"),]
                                        )
                                    est = exp['automl'](**estimator_params)

                                print('Start est fit')
                                start = time.time()
                                est.fit(impute_train, y_train)
                                stop = time.time()
                                duration = stop - start
                                print('Fitted')
                                if exp['automl'] is tpot2.TPOTClassifier:
                                    est.classes_ = est.fitted_pipeline_.classes_
                                print(est.fitted_pipeline_)
                                print('score start')
                                train_score = score(est, impute_train, y_train, r_or_c=r_or_c)
                                print('train score:', train_score)
                                ori_test_score = score(est, X_test, y_test, r_or_c=r_or_c)
                                print('original test score:', ori_test_score)
                                imputed_test_score = score(est, impute_test, y_test, r_or_c=r_or_c)
                                print('imputed test score:', imputed_test_score)
                                print('score end')
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                all_scores['train_score'] = train_score
                                all_scores['ori_test_score']=ori_test_score
                                all_scores['imputed_test_score'] = imputed_test_score
                                all_scores["start"] = start
                                all_scores["taskid"] = taskid
                                all_scores["level"] = level
                                all_scores["type"] = type_1
                                all_scores["exp_name"] = 'Imputed_Predictive_Capacity'
                                all_scores["name"] = openml.datasets.get_dataset(taskid).name
                                all_scores["duration"] = duration
                                all_scores["run"] = num_runs
                                all_scores["fit_model"] = est.fitted_pipeline_
                                all_scores["r_or_c"] = r_or_c

                                if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
                                    with open(f"{save_folder}/est_evaluated_individuals.pkl", "wb") as f:
                                        pickle.dump(est.evaluated_individuals, f)
                                        print('estimator working as intended')
                                print('check intended')
                                with open(f"{save_folder}/est_fitted_pipeline.pkl", "wb") as f:
                                    pickle.dump(est.fitted_pipeline_, f)

                                with open(f"{save_folder}/all_scores.pkl", "wb") as f:
                                    pickle.dump(all_scores, f)

                                print('EXP2 Finished')
                            

                                print("running experiment 3/3 - What is the best automl settings?")

        
                                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                                tpot_space = exp['automl'](**exp['params'])
                                print(exp['automl'])
                                print('Start tpot fit')
                                start = time.time()
                                tpot_space.fit(X_train_n, y_train)
                                stop = time.time()
                                duration = stop - start
                                print('Fitted')
                                if exp['automl'] is tpot2.TPOTClassifier:
                                    tpot_space.classes_ = tpot_space.fitted_pipeline_.classes_
                                print(tpot_space.fitted_pipeline_)
                                X_train_transform = tpot_space.fitted_pipeline_[0].transform(X_train_M)
                                print('transform worked')
                                rmse_loss_train3 = autoimpute.rmse_loss(ori_data=X_train, imputed_data=X_train_transform, data_m=np.multiply(mask_train.to_numpy(),1))
                                print('try transform')
                                X_test_transform = tpot_space.fitted_pipeline_[0].transform(X_test_M)
                                print('transform worked')
                                rmse_loss_test3 = autoimpute.rmse_loss(ori_data=X_test, imputed_data=X_test_transform, data_m=np.multiply(mask_test.to_numpy(),1))
                                print('score start')
                                train_score = score(tpot_space, X_train_n, y_train, r_or_c=r_or_c)
                                print('train score:', train_score)
                                test_score = score(tpot_space, X_test_n, y_test, r_or_c=r_or_c)
                                print('test score:', test_score)
                                ori_test_score = score(est, X_test, y_test, r_or_c=r_or_c)
                                print('original test score:', ori_test_score)
                                print('score end')
                                tpot_space_scores = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                
                                tpot_space_scores['train_score'] = train_score
                                tpot_space_scores['test_score']=test_score    
                                tpot_space_scores['ori_test_score']=ori_test_score    
                                tpot_space_scores["start"] = start
                                tpot_space_scores["taskid"] = taskid
                                tpot_space_scores["exp_name"] = exp['exp_name']
                                tpot_space_scores["name"] = openml.datasets.get_dataset(taskid).name
                                tpot_space_scores["duration"] = duration
                                tpot_space_scores["run"] = num_runs
                                tpot_space_scores["fit_model"] = tpot_space.fitted_pipeline_
                                tpot_space_scores["r_or_c"] = r_or_c
                                tpot_space_scores["rmse_loss_train3"] = rmse_loss_train3
                                tpot_space_scores["rmse_loss_test3"] = rmse_loss_test3


                                if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
                                    with open(f"{save_folder}/tpot_space_evaluated_individuals.pkl", "wb") as f:
                                        pickle.dump(tpot_space.evaluated_individuals, f)

                                with open(f"{save_folder}/tpot_space_fitted_pipeline.pkl", "wb") as f:
                                    pickle.dump(tpot_space.fitted_pipeline_, f)

                                with open(f"{save_folder}/tpot_space_scores.pkl", "wb") as f:
                                    pickle.dump(tpot_space_scores, f)
                                
                                #return
                                
                            except Exception as e:
                                trace =  traceback.format_exc() 
                                pipeline_failure_dict = {"taskid": taskid, "exp_name": exp['exp_name'], "run": num_runs, "error": str(e), "trace": trace, "level": level, "type": type_1}
                                print("failed on ")
                                print(save_folder)
                                print(e)
                                print(trace)

                                with open(f"{save_folder}/failed.pkl", "wb") as f:
                                    pickle.dump(pipeline_failure_dict, f)

                                return
                
        print(taskid)
        print('finished')
    print("all finished")
    return


### Additional Stuff GKetron Added
def add_missing(X, add_missing = 0.05, missing_type = 'MAR'):
    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X)
    missing_mask = X
    missing_mask = missing_mask.mask(missing_mask.isna(), True)
    missing_mask = missing_mask.mask(missing_mask.notna(), False)
    X = X.mask(X.isna(), 0)
    T = torch.tensor(X.to_numpy(), dtype=torch.float32)

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
        mask = (torch.rand(X.shape, dtype=torch.float32) < p).double()
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
        mask = torch.zeros(n, d, dtype=torch.float32).bool()
        num_no_missing = max(int(p_obs * d), 1)
        num_missing = d - num_no_missing
        obs_samples = np.random.choice(d, num_no_missing, replace=False)
        copy_samples = np.array([i for i in range(d) if i not in obs_samples])
        len_obs = len(obs_samples)
        len_na = len(copy_samples)
        coeffs = torch.randn(len_obs, len_na, dtype=torch.float32).double()
        Wx = X[:, obs_samples].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True, dtype=torch.float32)
        coeffs.double()
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na, dtype=torch.float32)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, obs_samples].mv(coeffs[:, j]) + x, dtype=torch.float32).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, obs_samples].mm(coeffs) + intercepts, dtype=torch.float32)
        ber = torch.rand(n, len_na, dtype=torch.float32)
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
        to_torch = torch.is_tensor(X, dtype=torch.float32) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = torch.from_numpy(X, dtype=torch.float32)
        mask = torch.zeros(n, d, dtype=torch.float32).bool() if to_torch else np.zeros((n, d)).astype(bool)
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
        coeffs = torch.randn(len_obs, len_na, dtype=torch.float32).double()
        Wx = X[:, idxs_params].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True, dtype=torch.float32)
        coeffs.double()
        ### Pick the intercepts to have a desired amount of missing values
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na, dtype=torch.float32)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, idxs_params].mv(coeffs[:, j]) + x, dtype=torch.float32).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts, dtype=torch.float32)
        ber = torch.rand(n, d_na, dtype=torch.float32)
        mask[:, idxs_nas] = ber < ps
        ## If the inputs of the logistic model are excluded from MNAR missingness,
        ## mask some values used in the logistic model at random.
        ## This makes the missingness of other variables potentially dependent on masked values
        if exclude_inputs:
            mask[:, idxs_params] = torch.rand(n, d_params, dtype=torch.float32) < p
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out
