import tpot2
import dill as pickle
import pandas as pd

for taskid in ['197']:
    fileoutput = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'
    csvout = pd.DataFrame(columns=['DatasetID','Exp_Name','Condition', 'Level', 'Triplicate','Exp1ImputeRMSE','Exp2ImputeModel','Exp2train_explained_var','Exp2train_r2', 
                                            'Exp2train_rmse', 'Exp2ori_explained_var', 'Exp2ori_r2', 'Exp2ori_rmse', 'Exp2impute_explained_var', 'Exp2impute_r2', 
                                            'Exp2impute_rmse', 'Exp2impute_pipe', 'Exp2duration', 'Exp3train_explained_var', 'Exp3train_r2', 'Exp3train_rmse', 'Exp3ori_explained_var', 
                                            'Exp3ori_r2', 'Exp3ori_rmse', 'Exp3impute_explained_var', 'Exp3impute_r2', 'Exp3impute_rmse',  
                                            'Exp3impute_pipe', 'Exp3duration'],
                            index=['/'+taskid+'_reg_simple_MAR_0.01_1_1/','/'+taskid+'_reg_simple_MAR_0.1_1/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_1_1/','/'+taskid+'_reg_simple_MAR_0.5_1/',
                                     '/'+taskid+'_reg_simple_MAR_0.9_1/','/'+taskid+'_reg_simple_MNAR_0.01_1/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_1/','/'+taskid+'_reg_simple_MNAR_0.3_1/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_1/', '/'+taskid+'_reg_simple_MNAR_0.9_1/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_1/','/'+taskid+'_reg_simple_MCAR_0.1_1/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_1/','/'+taskid+'_reg_simple_MCAR_0.5_1/',
                                    '/'+taskid+'_reg_simple_MCAR_0.9_1/','/'+taskid+'_reg_full_MAR_0.01_1/','/'+taskid+'_reg_full_MAR_0.1_1/',
                                   '/'+taskid+'_reg_full_MAR_0.3_1/','/'+taskid+'_reg_full_MAR_0.5_1/',
                                     '/'+taskid+'_reg_full_MAR_0.9_1/','/'+taskid+'_reg_full_MNAR_0.01_1/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_1/','/'+taskid+'_reg_full_MNAR_0.3_1/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_1/', '/'+taskid+'_reg_full_MNAR_0.9_1/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_1/','/'+taskid+'_reg_full_MCAR_0.1_1/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_1/','/'+taskid+'_reg_full_MCAR_0.5_1/', '/'+taskid+'_reg_full_MCAR_0.9_1/',
                                     '/'+taskid+'_reg_simple_MAR_0.01_2/','/'+taskid+'_reg_simple_MAR_0.1_2/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_1_2/','/'+taskid+'_reg_simple_MAR_0.5_2/',
                                     '/'+taskid+'_reg_simple_MAR_0.9_2/','/'+taskid+'_reg_simple_MNAR_0.01_2/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_2/','/'+taskid+'_reg_simple_MNAR_0.3_2/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_2/', '/'+taskid+'_reg_simple_MNAR_0.9_2/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_2/','/'+taskid+'_reg_simple_MCAR_0.1_2/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_2/','/'+taskid+'_reg_simple_MCAR_0.5_2/',
                                    '/'+taskid+'_reg_simple_MCAR_0.9_2/','/'+taskid+'_reg_full_MAR_0.01_2/','/'+taskid+'_reg_full_MAR_0.1_2/',
                                   '/'+taskid+'_reg_full_MAR_0.3_2/','/'+taskid+'_reg_full_MAR_0.5_2/',
                                     '/'+taskid+'_reg_full_MAR_0.9_2/','/'+taskid+'_reg_full_MNAR_0.01_2/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_2/','/'+taskid+'_reg_full_MNAR_0.3_2/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_2/', '/'+taskid+'_reg_full_MNAR_0.9_2/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_2/','/'+taskid+'_reg_full_MCAR_0.1_2/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_2/','/'+taskid+'_reg_full_MCAR_0.5_2/', '/'+taskid+'_reg_full_MCAR_0.9_2/',
                                     '/'+taskid+'_reg_simple_MAR_0.01_1_3/','/'+taskid+'_reg_simple_MAR_0.1_3/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_1_3/','/'+taskid+'_reg_simple_MAR_0.5_3/',
                                     '/'+taskid+'_reg_simple_MAR_0.9_3/','/'+taskid+'_reg_simple_MNAR_0.01_3/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_3/','/'+taskid+'_reg_simple_MNAR_0.3_3/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_3/', '/'+taskid+'_reg_simple_MNAR_0.9_3/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_3/','/'+taskid+'_reg_simple_MCAR_0.1_3/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_3/','/'+taskid+'_reg_simple_MCAR_0.5_3/',
                                    '/'+taskid+'_reg_simple_MCAR_0.9_3/','/'+taskid+'_reg_full_MAR_0.01_3/','/'+taskid+'_reg_full_MAR_0.1_3/',
                                   '/'+taskid+'_reg_full_MAR_0.3_3/','/'+taskid+'_reg_full_MAR_0.5_3/',
                                     '/'+taskid+'_reg_full_MAR_0.9_3/','/'+taskid+'_reg_full_MNAR_0.01_3/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_3/','/'+taskid+'_reg_full_MNAR_0.3_3/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_3/', '/'+taskid+'_reg_full_MNAR_0.9_3/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_3/','/'+taskid+'_reg_full_MCAR_0.1_3/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_3/','/'+taskid+'_reg_full_MCAR_0.5_3/', '/'+taskid+'_reg_full_MCAR_0.9_3/',
                                     ])
    #print(csvout)
    for exp in ['/reg_full_','/reg_simple_']:
        for item in ['MAR_', 'MCAR_', 'MNAR_']:
            for lvl in ['0.01_', '0.1_', '0.3_', '0.5_']:
                for iter in ['1/', '2/', '3/']:
                    normalpath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'+ taskid + exp + item + lvl + iter
                    imputepath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'+ taskid + exp + item + lvl + iter
                
                    try:
                        with open(normalpath + 'all_scores.pkl', 'rb') as file:
                            est = pickle.load(file)
                            #print(est)
                        with open(normalpath + 'est_fitted_pipeline.pkl', 'rb') as file:
                            est_pipeline = pickle.load(file)
                        with open(imputepath + 'tpot_space_scores.pkl', 'rb') as file:
                            tpot_space = pickle.load(file)
                            #print(tpot_space)
                        with open(imputepath + 'tpot_space_fitted_pipeline.pkl', 'rb') as file:
                            tpot_space_pipeline = pickle.load(file)
                            print(tpot_space_pipeline)
                        '''
                        csvout.loc[exp+item+lvl+iter] = pd.Series({'DatasetID':taskid,'Exp_Name': exp,'Condition': item, 'Level': lvl, 'Triplicate': iter,'Exp1ImputeRMSE': est['impute_rmse'] ,'Exp2ImputeModel': str(est['impute_space']),'Exp2train_auroc': est['train_score']['train_auroc'],
                                                                'Exp2train_acc': est['train_score']['train_accuracy'], 'Exp2train_bal_acc': est['train_score']['train_balanced_accuracy'], 
                                                                'Exp2train_logloss': est['train_score']['train_logloss'], 'Exp2test_auroc': est['ori_test_score']['auroc'], 'Exp2test_acc': est['ori_test_score']['accuracy'],
                                                            'Exp2test_bal_acc': est['ori_test_score']['balanced_accuracy'], 'Exp2test_logloss': est['ori_test_score']['logloss'], 'Exp2impute_auroc': est['imputed_test_score']['auroc'],
                                                                'Exp2impute_acc': est['imputed_test_score']['accuracy'], 'Exp2impute_bal_acc': est['imputed_test_score']['balanced_accuracy'], 'Exp2impute_logloss': est['imputed_test_score']['logloss'],
                                                                'Exp2impute_pipe': est_pipeline, 'Exp2duration': est['duration'], 'Exp3ImputeModel': tpot_space_pipeline, 'Exp3train_auroc': tpot_space['train_score']['train_auroc'],
                                                                    'Exp3train_acc': tpot_space['train_score']['train_accuracy'], 'Exp3train_bal_acc': tpot_space['train_score']['train_balanced_accuracy'], 'Exp3train_logloss': tpot_space['train_score']['train_logloss'],
                                                                    'Exp3impute_auroc': tpot_space['ori_test_score']['auroc'], 'Exp3impute_acc': tpot_space['ori_test_score']['accuracy'], 'Exp3impute_bal_acc': tpot_space['ori_test_score']['balanced_accuracy'], 'Exp3impute_logloss': tpot_space['ori_test_score']['logloss'], 'Exp3impute_pipe': tpot_space_pipeline, 'Exp3duration': tpot_space['duration']})
                        '''
                        csvout.loc[exp+item+lvl+iter] = pd.Series({'DatasetID':taskid,'Exp_Name': exp,'Condition': item, 'Level': lvl, 'Triplicate': iter,'Exp1ImputeRMSE': est['impute_rmse'] ,'Exp2ImputeModel': str(est['impute_space']),'Exp2train_explained_var': est['train_score']['explained_var'],'Exp2train_r2': est['train_score']['r2'], 
                                            'Exp2train_rmse': est['train_score']['rmse'], 'Exp2ori_explained_var': est['ori_test_score']['explained_var'], 'Exp2ori_r2': est['ori_test_score']['r2'], 
                                            'Exp2ori_rmse': est['ori_test_score']['rmse'], 'Exp2impute_explained_var': est['imputed_test_score']['explained_var'], 'Exp2impute_r2': est['imputed_test_score']['r2'], 
                                            'Exp2impute_rmse': est['imputed_test_score']['rmse'], 'Exp2impute_pipe': est['fit_model'], 'Exp2duration': est['duration'], 
                                            'Exp3train_explained_var': tpot_space['train_score']['explained_var'], 'Exp3train_r2': tpot_space['train_score']['r2'], 'Exp3train_rmse': tpot_space['train_score']['rmse'], 
                                            'Exp3ori_explained_var': tpot_space['ori_test_score']['explained_var'], 'Exp3ori_r2': tpot_space['ori_test_score']['r2'], 'Exp3ori_rmse': tpot_space['ori_test_score']['rmse'],
                                            'Exp3impute_explained_var': tpot_space['test_score']['explained_var'], 'Exp3impute_r2': tpot_space['test_score']['r2'], 'Exp3impute_rmse': tpot_space['test_score']['rmse'],  
                                            'Exp3impute_pipe': tpot_space['fit_model'], 'Exp3duration': tpot_space['duration']})
                        
                        print(taskid+item+lvl+iter+' passed')

                    except:
                        print(taskid+item+lvl+iter+' failed')
            
    output = csvout.to_csv(fileoutput+taskid+'.csv')
    print(taskid + 'complete')
print('all csvs complete')