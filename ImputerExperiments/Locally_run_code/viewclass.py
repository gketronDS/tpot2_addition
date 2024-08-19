import tpot2
import dill as pickle
import pandas as pd
import traceback

for taskid in ['6', '30']:
    fileoutput = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/'
    csvout = pd.DataFrame(columns=['DatasetID','Exp_Name','Condition', 'Level', 'Triplicate','Exp1ImputeRMSEAcc','Exp2ImputeModel','Exp2train_auroc','Exp2train_accuracy', 
                                            'Exp2train_balanced_accuracy', 'Exp2train_logloss','Exp2train_f1', 'Exp2ori_auroc','Exp2ori_accuracy', 
                                            'Exp2ori_balanced_accuracy', 'Exp2ori_logloss', 'Exp2ori_f1', 'Exp2impute_auroc','Exp2impute_accuracy', 
                                            'Exp2impute_balanced_accuracy', 'Exp2impute_logloss', 'Exp2impute_f1', 'Exp2ClassifierModel', 'Exp2duration', 
                                            'Exp3train_auroc', 'Exp3train_accuracy', 'Exp3train_balanced_accuracy', 
                                            'Exp3train_logloss', 'Exp3train_f1', 'Exp3ori_auroc', 'Exp3ori_accuracy', 'Exp3ori_balanced_accuracy', 
                                            'Exp3ori_logloss', 'Exp3ori_f1', 'Exp3impute_auroc', 'Exp3impute_accuracy', 'Exp3impute_balanced_accuracy', 
                                            'Exp3impute_logloss', 'Exp3impute_f1',
                                            'Exp3ImputeModel', 'Exp3ImputeRMSEAcc', 'Exp3ClassifierModel', 'Exp3duration'],
                            index=['/'+taskid+'_class_simple_MAR_0.01_1_1/','/'+taskid+'_class_simple_MAR_0.1_1/',
                                   '/'+taskid+'_class_simple_MAR_0.3_1_1/','/'+taskid+'_class_simple_MAR_0.5_1/',
                                     '/'+taskid+'_class_simple_MNAR_0.01_1/',
                                     '/'+taskid+'_class_simple_MNAR_0.1_1/','/'+taskid+'_class_simple_MNAR_0.3_1/',
                                     '/'+taskid+'_class_simple_MNAR_0.5_1/',
                                     '/'+taskid+'_class_simple_MCAR_0.01_1/','/'+taskid+'_class_simple_MCAR_0.1_1/',
                                     '/'+taskid+'_class_simple_MCAR_0.3_1/','/'+taskid+'_class_simple_MCAR_0.5_1/',
                                    '/'+taskid+'_class_full_MAR_0.01_1/','/'+taskid+'_class_full_MAR_0.1_1/',
                                   '/'+taskid+'_class_full_MAR_0.3_1/','/'+taskid+'_class_full_MAR_0.5_1/',
                                     '/'+taskid+'_class_full_MNAR_0.01_1/',
                                     '/'+taskid+'_class_full_MNAR_0.1_1/','/'+taskid+'_class_full_MNAR_0.3_1/',
                                     '/'+taskid+'_class_full_MNAR_0.5_1/',
                                     '/'+taskid+'_class_full_MCAR_0.01_1/','/'+taskid+'_class_full_MCAR_0.1_1/',
                                     '/'+taskid+'_class_full_MCAR_0.3_1/','/'+taskid+'_class_full_MCAR_0.5_1/',
                                     '/'+taskid+'_class_simple_MAR_0.01_2/','/'+taskid+'_class_simple_MAR_0.1_2/',
                                   '/'+taskid+'_class_simple_MAR_0.3_1_2/','/'+taskid+'_class_simple_MAR_0.5_2/',
                                     '/'+taskid+'_class_simple_MNAR_0.01_2/',
                                     '/'+taskid+'_class_simple_MNAR_0.1_2/','/'+taskid+'_class_simple_MNAR_0.3_2/',
                                     '/'+taskid+'_class_simple_MNAR_0.5_2/',
                                     '/'+taskid+'_class_simple_MCAR_0.01_2/','/'+taskid+'_class_simple_MCAR_0.1_2/',
                                     '/'+taskid+'_class_simple_MCAR_0.3_2/','/'+taskid+'_class_simple_MCAR_0.5_2/',
                                    '/'+taskid+'_class_full_MAR_0.01_2/','/'+taskid+'_class_full_MAR_0.1_2/',
                                   '/'+taskid+'_class_full_MAR_0.3_2/','/'+taskid+'_class_full_MAR_0.5_2/',
                                     '/'+taskid+'_class_full_MNAR_0.01_2/',
                                     '/'+taskid+'_class_full_MNAR_0.1_2/','/'+taskid+'_class_full_MNAR_0.3_2/',
                                     '/'+taskid+'_class_full_MNAR_0.5_2/',
                                     '/'+taskid+'_class_full_MCAR_0.01_2/','/'+taskid+'_class_full_MCAR_0.1_2/',
                                     '/'+taskid+'_class_full_MCAR_0.3_2/','/'+taskid+'_class_full_MCAR_0.5_2/',
                                     '/'+taskid+'_class_simple_MAR_0.01_1_3/','/'+taskid+'_class_simple_MAR_0.1_3/',
                                   '/'+taskid+'_class_simple_MAR_0.3_1_3/','/'+taskid+'_class_simple_MAR_0.5_3/',
                                     '/'+taskid+'_class_simple_MNAR_0.01_3/',
                                     '/'+taskid+'_class_simple_MNAR_0.1_3/','/'+taskid+'_class_simple_MNAR_0.3_3/',
                                     '/'+taskid+'_class_simple_MNAR_0.5_3/',
                                     '/'+taskid+'_class_simple_MCAR_0.01_3/','/'+taskid+'_class_simple_MCAR_0.1_3/',
                                     '/'+taskid+'_class_simple_MCAR_0.3_3/','/'+taskid+'_class_simple_MCAR_0.5_3/',
                                    '/'+taskid+'_class_full_MAR_0.01_3/','/'+taskid+'_class_full_MAR_0.1_3/',
                                   '/'+taskid+'_class_full_MAR_0.3_3/','/'+taskid+'_class_full_MAR_0.5_3/', '/'+taskid+'_class_full_MNAR_0.01_3/',
                                     '/'+taskid+'_class_full_MNAR_0.1_3/','/'+taskid+'_class_full_MNAR_0.3_3/',
                                     '/'+taskid+'_class_full_MNAR_0.5_3/',
                                     '/'+taskid+'_class_full_MCAR_0.01_3/','/'+taskid+'_class_full_MCAR_0.1_3/',
                                     '/'+taskid+'_class_full_MCAR_0.3_3/','/'+taskid+'_class_full_MCAR_0.5_3/',
                                     ])
    #print(csvout)
    for exp in ['class_full_','class_simple_']:
        for item in ['MAR_', 'MCAR_', 'MNAR_']:
            for lvl in ['0.01_', '0.1_', '0.3_', '0.5_']:
                for iter in ['1/', '2/', '3/']:
                    normalpath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/'+ taskid +'/'+exp + item + lvl + iter
                    imputepath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/'+ taskid +'/'+exp + item + lvl + iter
                
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
                            #print(tpot_space_pipeline)
                        
                        csvout.loc['/'+taskid+'_'+exp+item+lvl+iter] = pd.Series({'DatasetID':taskid,'Exp_Name': exp,'Condition': item, 'Level': lvl, 'Triplicate': iter,'Exp1ImputeRMSEAcc': est['impute_rmse'] ,'Exp2ImputeModel': str(est['impute_space']['model_name']),'Exp2train_auroc': est['train_score']['train_auroc'],'Exp2train_accuracy': est['train_score']['train_accuracy'], 
                                            'Exp2train_balanced_accuracy': est['train_score']['train_balanced_accuracy'], 'Exp2train_logloss': est['train_score']['train_logloss'],'Exp2train_f1': est['train_score']['train_f1'], 'Exp2ori_auroc': est['ori_test_score']['auroc'],'Exp2ori_accuracy': est['ori_test_score']['accuracy'], 
                                            'Exp2ori_balanced_accuracy': est['ori_test_score']['balanced_accuracy'], 'Exp2ori_logloss': est['ori_test_score']['logloss'],'Exp2ori_f1': est['ori_test_score']['f1'], 'Exp2impute_auroc': est['imputed_test_score']['auroc'],'Exp2impute_accuracy': est['imputed_test_score']['accuracy'], 
                                            'Exp2impute_balanced_accuracy': est['imputed_test_score']['balanced_accuracy'], 'Exp2impute_logloss': est['imputed_test_score']['logloss'],'Exp2impute_f1': est['imputed_test_score']['f1'], 'Exp2ClassifierModel': str(est['fit_model'][0]).split('(')[0], 'Exp2duration': est['duration'], 
                                            'Exp3train_auroc': tpot_space['train_score']['train_auroc'], 'Exp3train_accuracy': tpot_space['train_score']['train_accuracy'], 'Exp3train_balanced_accuracy': tpot_space['train_score']['train_balanced_accuracy'], 
                                            'Exp3train_logloss': tpot_space['train_score']['train_logloss'], 'Exp3train_f1': tpot_space['train_score']['train_f1'], 'Exp3ori_auroc': tpot_space['ori_test_score']['auroc'], 'Exp3ori_accuracy': tpot_space['ori_test_score']['accuracy'], 'Exp3ori_balanced_accuracy': tpot_space['ori_test_score']['balanced_accuracy'], 
                                            'Exp3ori_logloss': tpot_space['ori_test_score']['logloss'], 'Exp3ori_f1': tpot_space['ori_test_score']['f1'], 'Exp3impute_auroc': tpot_space['test_score']['auroc'], 'Exp3impute_accuracy': tpot_space['test_score']['accuracy'], 'Exp3impute_balanced_accuracy': tpot_space['test_score']['balanced_accuracy'], 
                                            'Exp3impute_logloss': tpot_space['test_score']['logloss'], 'Exp3impute_f1': tpot_space['test_score']['f1'],
                                            'Exp3ImputeModel': str(tpot_space['fit_model'][0]).split('(')[0], 'Exp3ImputeRMSEAcc': tpot_space["rmse_loss_test3"] ,'Exp3ClassifierModel': str(tpot_space['fit_model'][1]).split('(')[0] ,'Exp3duration': tpot_space['duration']})
                        
                        print(taskid+exp+item+lvl+iter+' passed')

                    except Exception as e:
                        print(taskid+exp+item+lvl+iter+' failed')
                        trace =  traceback.format_exc()
                        print(e)
                        print(trace) 
            
    output = csvout.to_csv(fileoutput+taskid+'.csv')
    print(taskid + 'complete')
print('all csvs complete')