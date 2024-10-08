import tpot2
import dill as pickle
import pandas as pd
import traceback

dflist=[]
#'189', '197', '198', '215', '216', '218', '1193', '1199', '1200', '1213', '42183', '42225', '42712',  '287', '42688'
for taskid in ['189', '197', '198', '215', '216', '218', '1193', '1199', '1200', '1213', '42183', '42545', '42225', '42712', '287', '42688']:
    fileoutput = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'
    csvout = pd.DataFrame(columns=['DatasetID','Exp_Name','Condition', 'Level', 'Triplicate','Exp1ImputeRMSEAcc','Exp2ImputeModel','Exp2train_explained_var','Exp2train_r2', 
                                            'Exp2train_rmse', 'Exp2ori_explained_var', 'Exp2ori_r2', 'Exp2ori_rmse', 'Exp2impute_explained_var', 'Exp2impute_r2', 
                                            'Exp2impute_rmse', 'Exp2RegressorModel', 'Exp2duration','Exp2inference_duration', 'Exp3train_explained_var', 'Exp3train_r2', 'Exp3train_rmse', 'Exp3ori_explained_var', 
                                            'Exp3ori_r2', 'Exp3ori_rmse', 'Exp3impute_explained_var', 'Exp3impute_r2', 'Exp3impute_rmse',  
                                            'Exp3ImputeModel','Exp3ImputeRMSEAcc','Exp3RegressorModel','Exp3duration', 'Exp3inference_duration'],
                            index=['/'+taskid+'_reg_simple_MAR_0.01_1/','/'+taskid+'_reg_simple_MAR_0.1_1/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_1/','/'+taskid+'_reg_simple_MAR_0.5_1/',
                                     '/'+taskid+'_reg_simple_MNAR_0.01_1/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_1/','/'+taskid+'_reg_simple_MNAR_0.3_1/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_1/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_1/','/'+taskid+'_reg_simple_MCAR_0.1_1/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_1/','/'+taskid+'_reg_simple_MCAR_0.5_1/',
                                    '/'+taskid+'_reg_full_MAR_0.01_1/','/'+taskid+'_reg_full_MAR_0.1_1/',
                                   '/'+taskid+'_reg_full_MAR_0.3_1/','/'+taskid+'_reg_full_MAR_0.5_1/',
                                     '/'+taskid+'_reg_full_MNAR_0.01_1/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_1/','/'+taskid+'_reg_full_MNAR_0.3_1/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_1/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_1/','/'+taskid+'_reg_full_MCAR_0.1_1/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_1/','/'+taskid+'_reg_full_MCAR_0.5_1/',
                                     '/'+taskid+'_reg_simple_MAR_0.01_2/','/'+taskid+'_reg_simple_MAR_0.1_2/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_2/','/'+taskid+'_reg_simple_MAR_0.5_2/',
                                     '/'+taskid+'_reg_simple_MNAR_0.01_2/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_2/','/'+taskid+'_reg_simple_MNAR_0.3_2/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_2/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_2/','/'+taskid+'_reg_simple_MCAR_0.1_2/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_2/','/'+taskid+'_reg_simple_MCAR_0.5_2/',
                                    '/'+taskid+'_reg_full_MAR_0.01_2/','/'+taskid+'_reg_full_MAR_0.1_2/',
                                   '/'+taskid+'_reg_full_MAR_0.3_2/','/'+taskid+'_reg_full_MAR_0.5_2/',
                                     '/'+taskid+'_reg_full_MNAR_0.01_2/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_2/','/'+taskid+'_reg_full_MNAR_0.3_2/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_2/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_2/','/'+taskid+'_reg_full_MCAR_0.1_2/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_2/','/'+taskid+'_reg_full_MCAR_0.5_2/',
                                     '/'+taskid+'_reg_simple_MAR_0.01_3/','/'+taskid+'_reg_simple_MAR_0.1_3/',
                                   '/'+taskid+'_reg_simple_MAR_0.3_3/','/'+taskid+'_reg_simple_MAR_0.5_3/',
                                     '/'+taskid+'_reg_simple_MNAR_0.01_3/',
                                     '/'+taskid+'_reg_simple_MNAR_0.1_3/','/'+taskid+'_reg_simple_MNAR_0.3_3/',
                                     '/'+taskid+'_reg_simple_MNAR_0.5_3/',
                                     '/'+taskid+'_reg_simple_MCAR_0.01_3/','/'+taskid+'_reg_simple_MCAR_0.1_3/',
                                     '/'+taskid+'_reg_simple_MCAR_0.3_3/','/'+taskid+'_reg_simple_MCAR_0.5_3/',
                                    '/'+taskid+'_reg_full_MAR_0.01_3/','/'+taskid+'_reg_full_MAR_0.1_3/',
                                   '/'+taskid+'_reg_full_MAR_0.3_3/','/'+taskid+'_reg_full_MAR_0.5_3/', '/'+taskid+'_reg_full_MNAR_0.01_3/',
                                     '/'+taskid+'_reg_full_MNAR_0.1_3/','/'+taskid+'_reg_full_MNAR_0.3_3/',
                                     '/'+taskid+'_reg_full_MNAR_0.5_3/',
                                     '/'+taskid+'_reg_full_MCAR_0.01_3/','/'+taskid+'_reg_full_MCAR_0.1_3/',
                                     '/'+taskid+'_reg_full_MCAR_0.3_3/','/'+taskid+'_reg_full_MCAR_0.5_3/',
                                     ])
    #print(csvout)
    for exp in ['reg_full_','reg_simple_']:
        for item in ['MAR_', 'MCAR_', 'MNAR_']:
            for lvl in ['0.01_', '0.1_', '0.3_', '0.5_']:
                for iter in ['1/', '2/', '3/']:
                    normalpath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'+ taskid +'/'+exp + item + lvl + iter
                    imputepath = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/'+ taskid +'/'+exp + item + lvl + iter
                
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
                        
                        csvout.loc['/'+taskid+'_'+exp+item+lvl+iter] = pd.Series({'DatasetID':taskid,'Exp_Name': exp,'Condition': item, 'Level': lvl, 'Triplicate': iter,'Exp1ImputeRMSEAcc': est['impute_rmse'] ,'Exp2ImputeModel': str(est['impute_space']['model_name']),'Exp2train_explained_var': est['train_score']['train_explained_var'],'Exp2train_r2': est['train_score']['train_r2'], 
                                            'Exp2train_rmse': est['train_score']['train_rmse'], 'Exp2ori_explained_var': est['ori_test_score']['explained_var'], 'Exp2ori_r2': est['ori_test_score']['r2'], 
                                            'Exp2ori_rmse': est['ori_test_score']['rmse'], 'Exp2impute_explained_var': est['imputed_test_score']['explained_var'], 'Exp2impute_r2': est['imputed_test_score']['r2'], 
                                            'Exp2impute_rmse': est['imputed_test_score']['rmse'], 'Exp2RegressorModel': str(est['fit_model'][0]).split('(')[0], 'Exp2duration': est['duration'],'Exp2inference_duration': est['inference_time'] ,
                                            'Exp3train_explained_var': tpot_space['train_score']['train_explained_var'], 'Exp3train_r2': tpot_space['train_score']['train_r2'], 'Exp3train_rmse': tpot_space['train_score']['train_rmse'], 
                                            'Exp3ori_explained_var': tpot_space['ori_test_score']['explained_var'], 'Exp3ori_r2': tpot_space['ori_test_score']['r2'], 'Exp3ori_rmse': tpot_space['ori_test_score']['rmse'],
                                            'Exp3impute_explained_var': tpot_space['test_score']['explained_var'], 'Exp3impute_r2': tpot_space['test_score']['r2'], 'Exp3impute_rmse': tpot_space['test_score']['rmse'],  
                                            'Exp3ImputeModel': str(tpot_space['fit_model'][0]).split('(')[0], 'Exp3ImputeRMSEAcc': tpot_space["rmse_loss_test3"] ,'Exp3RegressorModel': str(tpot_space['fit_model'][1]).split('(')[0] ,'Exp3duration': tpot_space['duration'], 'Exp3inference_duration': tpot_space['inference_time']})
                        
                        print(taskid+exp+item+lvl+iter+' passed')

                    except Exception as e:
                        print(taskid+exp+item+lvl+iter+' failed')
                        trace =  traceback.format_exc()
                        print(e)
                        print(trace) 
    dflist.append(csvout)
    print(taskid + 'complete')
result = pd.concat(dflist, ignore_index=True)
output = result.to_csv(fileoutput+'reg'+'.csv')
print('all csvs complete')