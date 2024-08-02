import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import time
import utils
import warnings
warnings.filterwarnings("ignore")
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from tpot2.builtin_modules.imputer import GainImputer


class AutoImputer():
  def __init__(self, internal_folds=10, n_trials=200, 
               random_state = None, CV_state = True, added_missing = 0.05, 
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
      my_params = trial_suggestion(trial, self.model_names, 
                                             column_len=X.shape[1],
                                             n_samples = X.shape[0], 
                                             random_state=self.random_state)
      trial.set_user_attr("out_params", my_params)
      my_model = MyModel(**my_params)
      return score(trial, splitting, my_model, X, 
                             self.missing_set_train, self.masked_set_train)
    self.study = optuna.create_study(sampler=self.sampler, 
                                     direction=self.direction)
    self.study.optimize(obj, n_trials=self.n_trials, 
                        n_jobs=self.n_jobs, 
                        show_progress_bar=self.show_progress, 
                        gc_after_trial=self.garbage_collect)
    self.best_model = MyModel(**self.study.best_trial.user_attrs["out_params"]) 
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
    return self.transform(X, y)
  
  def missing_sets(self):
    return self.missing_set_train, self.masked_set_train, self.missing_set_test, self.masked_set_test

def trial_suggestion(trial: optuna.trial.Trial, model_names,column_len, n_samples, random_state):
    model_name = trial.suggest_categorical('model_name', model_names)# Model names set up to run on multiple or individual models. Options include: 'SimpleImputer' , 'IterativeImputer','KNNImputer', 'VAEImputer', 'GAIN', 'Opt.SVM', 'Opt.Trees'.  Not yet working: 'DLImputer', Random Forest Imputer 
    my_params = {}
    match model_name:
      case 'SimpleImputer':
        my_params = params_SimpleImpute(trial)  #Takes data from each column to input a value for all missing data. 
      case 'IterativeImputer':
        my_params = params_IterativeImpute(trial, column_len,random_state) #Uses the dependence between columns in the data set to predict for the one column. Predictions occur through a variety of strategies.
      case 'KNNImputer':
        my_params = params_KNNImpute(trial, n_samples) #uses nearest neighbors to predict missing values with k-neighbors in n-dimensional space with known values.
      case 'GAIN':
        my_params = params_GAINImpute(trial, random_state) #Uses a generative adversarial network model to predict values. 
    my_params['model_name'] = model_name
    return my_params
  
def MyModel(**params):
    this_model = {}
    these_params = params
    model_name = these_params['model_name']
    del these_params['model_name']
    match model_name:
        case 'SimpleImputer':
            this_model = SimpleImputer(
                **these_params
                )
        case 'IterativeImputer':
            match params['estimator']:
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
            'initial_strategy' : params['initial_strategy'],
            'n_nearest_features' : params['n_nearest_features'],
            'imputation_order' : params['imputation_order'],
            }
            if 'sample_posterior' in params:
                final_params['sample_posterior'] = params['sample_posterior']
            if 'random_state' in params:
                final_params['random_state'] = params['random_state']
            this_model = IterativeImputer(
                **final_params
                )
        case 'KNNImputer':
            this_model = KNNImputer( 
                **these_params
                )
        case 'GAIN':
            this_model = GainImputer(
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

def params_SimpleImpute(trial):
    params = {}
    params['strategy'] = trial.suggest_categorical('strategy', ['mean', 'median', 'most_frequent', 'constant'])
    return params

def params_IterativeImpute(trial, column_len, random_state=None):
    params = {}
    params['estimator'] = trial.suggest_categorical('estimator', ['Bayesian', 'RFR', 'Ridge', 'KNN'])
    params['initial_strategy'] = trial.suggest_categorical('initial_strategy', ['mean', 'median', 'most_frequent', 'constant'])
    params['n_nearest_features'] = trial.suggest_int('n_nearest_features', 1, column_len)
    params['imputation_order'] = trial.suggest_categorical('imputation_order', ['ascending', 'descending', 'roman', 'arabic', 'random'])
    if params['estimator'] =='Bayesian':
        params['sample_posterior'] = trial.suggest_categorical('sample_posterior', [True, False])
    if random_state is not None:
        params['random_state'] = random_state
    return params

def params_KNNImpute(trial, n_samples):
    params = {}
    params['n_neighbors'] = trial.suggest_int('n_neighbors', 1, max(100, n_samples))
    params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])
    return params

def params_GAINImpute(trial, random_state=None):
    params ={ 
        'batch_size': trial.suggest_int('batch_size', 1, 1000, log=True),
        'hint_rate': trial.suggest_float('hint_rate', 0.01, 0.99),
        'alpha': trial.suggest_int('alpha', 0, 100),
        'iterations': trial.suggest_int('iterations', 1, 100000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
        'p_miss': trial.suggest_float('p_miss', 0.01, 0.3),
    }
    if random_state is not None: 
            params['random_state'] = random_state
    return params

