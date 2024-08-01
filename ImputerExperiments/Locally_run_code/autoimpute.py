import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import time
import utils
import autoutils
import warnings
warnings.filterwarnings("ignore")


class AutoImputer():
  def __init__(self, internal_folds=10, n_trials=200, 
               random_state = 42, CV_state = True, added_missing = 0.05, 
               missing_type: str = 'MAR', 
               model_names: list = ['SimpleImputer' , 
                                    'IterativeImputer',
                                    'KNNImputer', 'GAIN', 
                                    'RandomForestImputer'], 
              sampler = optuna.samplers.TPESampler(), 
              direction ='minimize', n_jobs = 1, 
              show_progress = True, garbage_collect=True):
    '''
    Loads in meta-parameters. 

    fname = str, contains relative path to csv file containing missingness set. 
    Combine after running missing_function to also get mask version of the 
    missing set. Needed for GAIN and VAE. 

    loss_function: function : set category to optimize for. options: cv-rmse, rmse

    meta_cv_folds: cross validation outside of the optimization function

    sampler_type: optuna optimization algorithm

    n_trials: number of trials in the parameter space

    random_state: sets reproducable sampling. 
    '''
    self.n_trials=n_trials
    self.random_state = random_state
    self.internal_folds = internal_folds
    self.acc_model = sklearn.linear_model.LogisticRegression()
    self.CV_state = CV_state
    self.added_missing = added_missing
    self.missing_type = missing_type
    self.model_names = model_names
    self.sampler = sampler
    self.direction = direction
    self.n_jobs = n_jobs
    self.show_progress = show_progress
    self.garbage_collect = garbage_collect

  def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
    start_time = time.time()
    self.missing_set_train, self.masked_set_train = utils.add_missing(X, 
                add_missing=self.added_missing, missing_type=self.missing_type)
    if self.CV_state == False:
      splitting = ShuffleSplit(n_splits=1, random_state=self.random_state)
    else:
      splitting = KFold(n_splits=self.internal_folds, 
                        random_state=self.random_state, shuffle=True)
    def obj(trial):
      my_params = autoutils.trial_suggestion(trial, self.model_names, 
                                             column_len=len(X.columns), 
                                             random_state=self.random_state)
      trial.set_user_attr("out_params", my_params)
      my_model = autoutils.MyModel(random_state = self.random_state, 
                                   **my_params)
      return autoutils.score(trial, splitting, my_model, X, 
                             self.missing_set_train, self.masked_set_train)
    self.study = optuna.create_study(sampler=self.sampler, 
                                     direction=self.direction)
    self.study.optimize(obj, n_trials=self.n_trials, 
                        n_jobs=self.n_jobs, 
                        show_progress_bar=self.show_progress, 
                        gc_after_trial=self.garbage_collect)
    self.best_model = autoutils.MyModel(random_state = self.random_state, 
                              **self.study.best_trial.user_attrs["out_params"]) 
    self.best_model.fit(X)
    stop_time = time.time()
    self.fit_time = stop_time - start_time
    
  def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
    transstart_time = time.time()
    #self.missing_set_test, self.masked_set_test = utils.add_missing
    # (X, add_missing=self.added_missing, missing_type=self.missing_type)
    imputed = self.best_model.transform(X)
    if 'numpy' in str(type(imputed)):
      imputed = pd.DataFrame(imputed, columns=X.columns.values)
    transstop_time = time.time()
    self.transform_time = transstop_time - transstart_time
    return imputed

  def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
    self.fit(X, y)
    imputed_set = self.transform(X, y)
    return imputed_set
  
  def missing_sets(self):
    return self.missing_set_train, self.masked_set_train, self.missing_set_test, self.masked_set_test
  
import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from param_grids import params_SimpleImpute, params_IterativeImpute, params_KNNImpute, params_RandomForestImpute, params_GAINImpute
from transformers import RandomForestImputer, GAINImputer


def trial_suggestion(trial: optuna.trial.Trial, model_names,column_len,random_state):
    model_name = trial.suggest_categorical('model_name', model_names)# Model names set up to run on multiple or individual models. Options include: 'SimpleImputer' , 'IterativeImputer','KNNImputer', 'VAEImputer', 'GAIN', 'Opt.SVM', 'Opt.Trees'.  Not yet working: 'DLImputer', Random Forest Imputer 
    my_params = {}
    match model_name:
      case 'SimpleImputer':
        my_params = params_SimpleImpute(trial, model_name)  #Takes data from each column to input a value for all missing data. 
      case 'IterativeImputer':
        my_params = params_IterativeImpute(trial,model_name) #Uses the dependence between columns in the data set to predict for the one column. Predictions occur through a variety of strategies.
      case 'KNNImputer':
        my_params = params_KNNImpute(trial,model_name) #uses nearest neighbors to predict missing values with k-neighbors in n-dimensional space with known values.
      case 'GAIN':
        my_params = params_GAINImpute(trial,model_name) #Uses a generative adversarial network model to predict values. 
      case 'RandomForestImputer': #uses a hyperparameter optimized SVM model to predict values to impute.
        my_params = params_RandomForestImpute(trial, model_name)
    my_params['model_name'] = model_name
    return my_params
  
def MyModel(random_state, **params):
    these_params = params
    model_name = these_params['model_name']
    del these_params['model_name']

    match model_name:
        case 'SimpleImputer':
            this_model = SimpleImputer(
                **these_params
                )
        case 'IterativeImputer':
            this_model = IterativeImputer(
                **these_params
                )
        case 'KNNImputer':
            this_model = KNNImputer( 
                **these_params
                )
        case 'GAIN':
            this_model = GAINImputer(
                **these_params
                )
        case 'RandomForestImputer': #uses a hyperparameter optimized SVM model to predict values to impute.
            this_model = RandomForestImputer(
                **these_params
            )
    return this_model

def score(trial: optuna.trial.Trial, splitting, my_model, X: pd.DataFrame, missing_set: pd.DataFrame, masked_set:pd.DataFrame):
    avg_cv_rmse = []
    for i, (train_index, test_index) in enumerate(splitting.split(X)):
        missing_train, missing_test, X_test, masked_test = missing_set.iloc[train_index], missing_set.iloc[test_index], X.iloc[test_index], masked_set.iloc[test_index]
        try: 
            my_model.fit(missing_train)
            imputed = my_model.transform(missing_test)

            if 'numpy' in str(type(imputed)):
                imputed = pd.DataFrame(imputed, columns=missing_set.columns.values)

            if imputed.isnull().sum().sum() > 0: 
                rmse_val = np.inf
                return rmse_val
            
            rmse_val = rmse_loss(ori_data=X_test.to_numpy(), imputed_data=imputed.to_numpy(), data_m=np.multiply(masked_test.to_numpy(),1))
            avg_cv_rmse.append(rmse_val)
        except:
            rmse_val = np.inf
            return rmse_val
    cv_rmse = sum(avg_cv_rmse)/float(len(avg_cv_rmse))
    trial.set_user_attr('cv-rmse', cv_rmse)
    return cv_rmse

def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data
    Args:
        - ori_data: original data without missing values
        - imputed_data: imputed data
        - data_m: indicator matrix for missingness    
    Returns:
        - rmse: Root Mean Squared Error
    '''
    #ori_data, norm_parameters = normalization(ori_data)
    #imputed_data, _ = normalization(imputed_data, norm_parameters)
    # Only for missing values
    nominator = np.nansum(((data_m) * ori_data - (data_m) * imputed_data)**2)
    #print(nominator)
    denominator = np.sum(data_m)
    rmse = np.sqrt(nominator/float(denominator))
    return rmse

""" BEYOND THIS POINT WRITTEN BY Piotr Slomka - https://www.nature.com/articles/s41598-021-93651-5.pdf"""

def convert_col(pd_series,keep_missing=True):
    df = pd.DataFrame(pd_series)
    new_df =  df.stack().str.get_dummies(sep=',')
    new_df.columns = new_df.columns.str.strip()
    if keep_missing and 'nan' in new_df.columns: #this means there were missing values
        new_df.loc[new_df['nan']==1] = np.nan
        new_df.drop('nan', inplace=True,axis=1)
    new_df = new_df.stack().groupby(level=[0,1,2]).sum().unstack(level=[1,2])
    return new_df

#One hot encodes all categorical features. (if continuous included, each value will be counted as a discrete category)
def convert_all_cols(df,features_to_one_hot,keep_missing=True):
    new_df = pd.concat([convert_col(df[feature],keep_missing) for feature in features_to_one_hot],axis=1)
    return pd.concat([df.drop(features_to_one_hot, axis = 1),new_df],axis=1)

def one_hot_encode(spect_df,keep_missing=True):
    categorical_feature_names = spect_df.select_dtypes(include=object).columns
    numerical_feature_names = spect_df.select_dtypes(include=float).columns
    spect_df = convert_all_cols(spect_df,categorical_feature_names,keep_missing=keep_missing)
    #change binary ints into floats
    spect_df = spect_df.astype(float)
    #change column names from tuples to strings for the imputer to work better
    change_column_name_from_tuple_to_string = lambda cname: '__'.join(cname) if type(cname)==tuple else cname
    spect_df.columns = spect_df.columns.map(change_column_name_from_tuple_to_string)
    return spect_df

import tpot2
import optuna
import sklearn

def params_SimpleImpute(trial, name=None):
    params = {}
    params['strategy'] = trial.suggest_categorical('strategy', ['mean', 'median', 'most_frequent', 'constant'])
    param_grid = {
        'strategy': params['strategy']
    }

    return param_grid

def params_IterativeImpute(trial, name=None):
    params = {}
    #params['estimator'] = trial.suggest_categorical('estimator', ['Bayesian', 'RFR', 'Ridge', 'KNN'])
    params['sample_posterior'] = trial.suggest_categorical('sample_posterior', [True, False])
    params['initial_strategy'] = trial.suggest_categorical('initial_strategy', ['mean', 'median', 'most_frequent', 'constant'])
    params['n_nearest_features'] = None
    params['imputation_order'] = trial.suggest_categorical('imputation_order', ['ascending', 'descending', 'roman', 'arabic', 'random'])
    
    est = params['estimator']
    match est:
        case 'Bayesian':
                estimator = sklearn.linear_model.BayesianRidge()
        case 'RFR':
                estimator = sklearn.ensemble.RandomForestRegressor()
        case 'Ridge':
                estimator = sklearn.linear_model.Ridge()
        case 'KNN':
                estimator = sklearn.neighbors.KNeighborsRegressor()
    
    final_params = {
            'estimator' : estimator,
            'sample_posterior' : params['sample_posterior'],
            'initial_strategy' : params['initial_strategy'],
            'n_nearest_features' : params['n_nearest_features'],
            'imputation_order' : params['imputation_order'],
    }

    if "random_state" in params:
        final_params['random_state'] = params['random_state']

    return final_params
    return param_grid

def params_KNNImpute(trial, name=None):
    params = {}
    #params['n_nearest_features'] = None
    params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])
    params['keep_empty_features'] = trial.suggest_categorical('keep_empty_features', [False])
    param_grid = {
        #'n_neighbors': params['n_nearest_features'],
        'weights': params['weights'],
        'add_indicator': False,
        'keep_empty_features': params['keep_empty_features'],
    }
    return param_grid

def params_RandomForestImpute(trial, name=None):
    params = {}
    params['max_iter'] = trial.suggest_int('max_iter', 1, 100, step = 1, log = True)
    params['decreasing'] = trial.suggest_categorical('decreasing', [True, False])
    params['n_estimators'] = trial.suggest_int('max_iter', 50, 200, step = 1, log=True)
    params['max_depth'] = None
    params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.0, 1.0, step = 0.1)
    params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.1, 0.9, step = 0.1)
    params['max_features'] = trial.suggest_float('max_features', 0.1, 0.9, step = 0.1)
    params['max_leaf_nodes'] = None
    #params['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
    #params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
    params['warm_start'] = trial.suggest_categorical('warm_start', [True, False])
    params['class_weight'] = None
    param_grid = {
        'max_iter': params['max_iter'],
        'decreasing': params['decreasing'],
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'], 
        'max_leaf_nodes': params['max_leaf_nodes'],
        #'bootstrap': params['bootstrap'],
        #'oob_score': params['oob_score'],
        'warm_start': params['warm_start'],
        'class_weight': params['class_weight']
    }
    return param_grid

def params_GAINImpute(trial, name=None):
    return { 
        'batch_size': trial.suggest_int('batch_size', 1, 1000, log=True),
        'hint_rate': trial.suggest_float('hint_rate', 0.01, 0.99, step = 0.01),
        'alpha': trial.suggest_int('alpha', 0, 100, step = 1),
        'iterations': trial.suggest_int('iterations', 1, 100000, log=True)
    }
