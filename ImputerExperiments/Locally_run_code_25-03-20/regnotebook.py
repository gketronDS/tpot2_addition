import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy
import re
import seaborn as sns
sns.set_theme()
pd.set_option('display.max_columns', None)

path = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg.csv'

reg_data = pd.read_csv(path)

reg_data.head(5)

reg_data = reg_data.dropna(how='any')

reg_data.head(5)

reg_data = reg_data.replace('_', '', regex=True)
reg_data = reg_data.replace('/', '', regex=True)
reg_data.head(5)
reg_data.drop(columns=reg_data.columns[0], axis=1, inplace=True)
convert_dict = {'DatasetID': int}
reg_data = reg_data.astype(convert_dict)
convert_dict = {'DatasetID': str}
reg_data = reg_data.astype(convert_dict)
reg_data.head(5)

reg_data = reg_data.sort_values(by=['DatasetID', 'Condition', 'Level', 'Triplicate'], ascending=True)
reg_data.head(-1)

#print(class_data[(class_data.Exp_Name == 'classfull') & (class_data.Level == '0.01')]['Exp2ImputeModel'].value_counts())

def display_model_proportions(df, exp, savepath, complex = False, dataset_list=None, show=False):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    if complex:
        name = 'regfull'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Complex'
        sub2 = "Impute First"
        
    else:
        name = 'regsimple'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Simple'
        sub2 = "Simple First"
        

    xvals = [0.01, 0.1, 0.3, 0.5]
    xlabel = 'Percent Missing (%)'
    ylabel = 'Percent of Time Selected (%)'

    all_models = {}
    mar_models = {}
    mcar_models = {}
    mnar_models = {}

    match exp:
        case 1: 
           pipe = 'Exp2ImputeModel'
           title = 'Imputer Models'
           subtitle = sub2
        case 2:
            pipe = 'Exp2RegressorModel'
            title = 'Regressor Models'
            subtitle = sub2
        case 3:
            pipe = 'Exp3ImputeModel'
            title = subtitle+' TPOT2 Imputer Models'
        case 4: 
            pipe = 'Exp3RegressorModel'
            title = subtitle+' TPOT2 Regressor Models'

    for model in temp[pipe].unique():
        new_list1 = []
        for val in xvals:
            try:
                new_list1.append(temp[temp.Level == str(val)][pipe].value_counts()[model]/temp[temp.Level == str(val)][pipe].value_counts().sum())
            except:
                new_list1.append(0.0)
        all_models[model] = new_list1
    for model in temp[temp.Condition == 'MAR'][pipe].unique():
        new_list2 = []
        for val in xvals:
            try:
                new_list2.append(temp[(temp.Condition == 'MAR')&(temp.Level == str(val))][pipe].value_counts()[model]/temp[(temp.Condition == 'MAR')&(temp.Level == str(val))][pipe].value_counts().sum())
            except:
                new_list2.append(0.0)
        mar_models[model] = new_list2
    for model in temp[temp.Condition == 'MCAR'][pipe].unique():
        new_list3 = []
        for val in xvals:
            try:
                new_list3.append(temp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][pipe].value_counts()[model]/temp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][pipe].value_counts().sum())
            except:
                new_list3.append(0.0)
        mcar_models[model] = new_list3
    for model in temp[temp.Condition == 'MNAR'][pipe].unique():
        new_list4 = []
        for val in xvals:
            try:
                new_list4.append(temp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][pipe].value_counts()[model]/temp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][pipe].value_counts().sum())
            except:
                new_list4.append(0.0)
        mnar_models[model] = new_list4
    fig, a = plt.subplots(2,2)
    for i, label in enumerate(all_models):
        a[0][0].plot(xvals,all_models[label], color="C"+str(i), label=str(label))
        try:
            a[0][1].plot(xvals,mar_models[label], color="C"+str(i))
        except:
            save = i
        try:
            a[1][0].plot(xvals,mcar_models[label], color="C"+str(i))
        except:
            save = i
        try:
            a[1][1].plot(xvals, mnar_models[label], color="C"+str(i))
        except:
            save = i
            
    a[0][0].set_title('All Conditions')
    a[0][0].set_xlabel(xlabel)
    a[0][0].set_ylabel(ylabel)
    a[0][0].set_xticks(np.arange(0, 0.6, 0.1))  
    a[0][0].set_yticks(np.arange(0, 1.1, 0.2))      
    a[0][1].set_title('Missing At Random')
    a[0][1].set_xlabel(xlabel)
    a[0][1].set_ylabel(ylabel)
    a[0][1].set_xticks(np.arange(0, 0.6, 0.1))  
    a[0][1].set_yticks(np.arange(0, 1.1, 0.2))  
    a[1][0].set_title('Missing Completely At Random')
    a[1][0].set_xlabel(xlabel)
    a[1][0].set_ylabel(ylabel)
    a[1][0].set_xticks(np.arange(0, 0.6, 0.1))  
    a[1][0].set_yticks(np.arange(0, 1.1, 0.2))  
    a[1][1].set_title('Missing Not At Random')
    a[1][1].set_xlabel(xlabel)
    a[1][1].set_ylabel(ylabel)
    a[1][1].set_xticks(np.arange(0, 0.6, 0.1))  
    a[1][1].set_yticks(np.arange(0, 1.1, 0.2))  
    fig.suptitle('Regression '+subtitle+' Model Space: '+ str(dataset_list)+' Selection Frequency of ' + title)
    lgd=fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    #fig.savefig(savepath + name+'_'+ str(dataset_list)+'_'+pipe+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    all_models['Missing_Fraction'] = xvals
    mar_models['Missing Fraction'] = xvals
    mcar_models['Missing Fraction'] = xvals
    mnar_models['Missing Fraction'] = xvals
    all_table = pd.DataFrame(all_models)
    mar_table = pd.DataFrame(mar_models)
    mcar_table = pd.DataFrame(mcar_models)
    mnar_table = pd.DataFrame(mnar_models)
    return all_table, mar_table, mcar_models, mnar_models

for twos in [True, False]:
    for i in range(1,5):
        complexed = twos
        all_table, mar_table, mcar_models, mnar_models = display_model_proportions(reg_data, exp=i, complex=complexed, savepath='/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/')
        #all_table= all_table.map('{:.0%}'.format)
        print(all_table)
        if complexed:
            name = 'regfull'
            #temp = temp[temp.Exp_Name == name]
            subtitle = 'Complex'
            sub2 = 'Impute First'
        else:
            name = 'regsimple'
            #temp = temp[temp.Exp_Name == name]
            subtitle = 'Simple'
            sub2 = 'Simple First'

        match i:
                case 1: 
                    pipe = 'Exp2ImputeModel'
                    title = 'Imputer_Models'
                    subtitle = sub2
                case 2:
                    pipe = 'Exp2RegressorModel'
                    title = 'Regressor_Models'
                    subtitle = sub2
                case 3:
                    pipe = 'Exp3ImputeModel'
                    title = subtitle+'_TPOT2_Imputer_Models'
                case 4: 
                    pipe = 'Exp3RegressorModel'
                    title = subtitle+'_TPOT2_Regressor_Models'
        all_table = all_table.T
        all_table['Total'] = all_table.mean(axis=1)
        all_table= all_table.map('{:.0%}'.format)
        all_table.columns = all_table.iloc[-1]
        all_table = all_table.add_suffix(" Missing")
        all_table.columns = [*all_table.columns[:-1], 'Total']
        all_table = all_table.drop('Missing_Fraction')
        out = all_table.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis/'+name+pipe+title+subtitle+str(i)+'.csv')

def display_scores_over_options(df, test, score_type, savepath,
                                dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'reg'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name != name+'full']
    name = 'Regression'
    match test:
        case 'Train':
            match score_type:
                case 'rmse':
                    imputer = 'Exp2train_rmse'
                    s_first = 'Exp2train_rmse'
                    complexer = 'Exp3train_rmse'
                    simpler = 'Exp3train_rmse'
                    ylabel = 'RMSE Score'
                case 'explained_var':
                    imputer = 'Exp2train_explained_var'
                    s_first = 'Exp2train_explained_var'
                    complexer = 'Exp3train_explained_var'
                    simpler = 'Exp3train_explained_var'
                    ylabel = 'Explained Variance (%)'
                case 'r2':
                    imputer = 'Exp2train_r2'
                    s_first = 'Exp2train_r2'
                    complexer = 'Exp3train_r2'
                    simpler = 'Exp3train_r2'
                    ylabel = r'$R_2$'
                case 'training_duration':
                    imputer = 'Exp2duration'
                    s_first = 'Exp2duration'
                    complexer = 'Exp3duration'
                    simpler = 'Exp3duration'
                    ylabel = 'Training Time (Seconds)'
                case 'RMSEAcc':
                    imputer = 'Exp1ImputeRMSEAcc'
                    s_first = 'Exp1ImputeRMSEAcc'
                    complexer = 'Exp3ImputeRMSEAcc'
                    simpler = 'Exp3ImputeRMSEAcc'
                    ylabel = 'Imputation RMSE'         
        case 'Test':
            match score_type:
                case 'rmse':
                    imputer = 'Exp2impute_rmse'
                    s_first = 'Exp2impute_rmse'
                    complexer = 'Exp3impute_rmse'
                    simpler = 'Exp3impute_rmse'
                    ylabel = 'RMSE Score'
                case 'explained_var':
                    imputer = 'Exp2impute_explained_var'
                    s_first = 'Exp2impute_explained_var'
                    complexer = 'Exp3impute_explained_var'
                    simpler = 'Exp3impute_explained_var'
                    ylabel = 'Explained Variance (%)'
                case 'r2':
                    imputer = 'Exp2impute_r2'
                    s_first = 'Exp2impute_r2'
                    complexer = 'Exp3impute_r2'
                    simpler = 'Exp3impute_r2'
                    ylabel = r'$R_2$'
                case 'training_duration':
                    imputer = 'Exp2inference_duration'
                    s_first = 'Exp2inference_duration'
                    complexer = 'Exp3inference_duration'
                    simpler = 'Exp3inference_duration'
                    ylabel = 'Inference Time (Seconds)'
                case 'RMSEAcc':
                    imputer = 'Exp1ImputeRMSEAcc'
                    s_first = 'Exp1ImputeRMSEAcc'
                    complexer = 'Exp3ImputeRMSEAcc'
                    simpler = 'Exp3ImputeRMSEAcc'
                    ylabel = 'Imputation RMSE'

    xvals = [0.01, 0.1, 0.3, 0.5]
    xlabel = 'Percent Missing'
    all_models = {}
    mar_models = {}
    mcar_models = {}
    mnar_models = {}


    for i, model in enumerate([imputer, complexer, simpler, s_first]):
        all_list = []
        for val in xvals:
            if i >= 2:
                try:
                    all_list.append(simpletemp[simpletemp.Level == str(val)][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[fulltemp.Level == str(val)][model].mean())
                except:
                    all_list.append(0.0)
        if i >= 2:
            all_models['simple_'+model] = all_list
        else:
            all_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler, s_first]):
        all_list = []
        for val in xvals:
            if i >= 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i >= 2:
            mar_models['simple_'+model] = all_list
        else:
            mar_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler, s_first]):
        all_list = []
        for val in xvals:
            if i >= 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i >= 2:           
            mcar_models['simple_'+model] = all_list
        else:
            mcar_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler, s_first]):
        all_list = []
        for val in xvals:
            if i >= 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i >= 2:
            mnar_models['simple_'+model] = all_list
        else:
            mnar_models[model] = all_list
    for sets in [all_models, mar_models, mcar_models, mnar_models]:
        sets['Impute First '+score_type] = sets[imputer]
        sets['Complex '+score_type] = sets[complexer]
        sets['Simple '+score_type] = sets['simple_'+simpler]
        sets['Simple First'+score_type] = sets['simple_'+s_first]
        del sets[imputer], sets[complexer], sets['simple_'+simpler], sets['simple_'+s_first]
    
    fig, a = plt.subplots(2,2,sharey=True, figsize=(12,10))
    fig.text(0.5, 0.93, f'{test} Split {ylabel}', transform=fig.transFigure, fontsize=16, ha='center')

    maxed = [0]
    for i, label in enumerate(all_models):
        a[0][0].plot(xvals,all_models[label], color="C"+str(i), label=str(label))
        maxed.append(max(all_models[label]))
        try:
            a[0][1].plot(xvals,mar_models[label], color="C"+str(i))
            maxed.append(max(mar_models[label]))
        except:
            save = i
        try:
            a[1][0].plot(xvals,mcar_models[label], color="C"+str(i))
            maxed.append(max(mcar_models[label]))
        except:
            save = i
        try:
            a[1][1].plot(xvals, mnar_models[label], color="C"+str(i))
            maxed.append(max(mnar_models[label]))
        except:
            save = i

    match score_type:
        case 'rmse':
            yaxes = np.arange(0, np.round(max(maxed)+np.round(max(maxed))/5, decimals=-3), np.round(max(maxed))/5)
        case 'explained_var':
            yaxes = np.arange(0, 1.2, 0.2)
        case 'r2':
            yaxes = np.arange(0, 1.2, 0.2)
        case 'training_duration':
            yaxes = np.arange(0, np.round(max(maxed))+np.round(max(maxed))/5, np.round(max(maxed))/5)
        case 'RMSEAcc':
            yaxes = np.arange(0, np.round(max(maxed), decimals=2)+np.round(max(maxed), decimals=2)/5, np.round(max(maxed), decimals=2)/5)
    
    #yaxes = np.arange(0, 8+0.5, 1)
    a[0][0].set_title('All Conditions')
    a[0][0].set_xlabel(xlabel)
    a[0][0].set_ylabel(ylabel)
    a[0][0].set_xticks(np.arange(0, 0.6, 0.1)) 
    a[0][0].set_yticks(yaxes)       
    
    a[0][1].set_title('Missing At Random')
    a[0][1].set_xlabel(xlabel)
    a[0][1].set_ylabel(ylabel)
    a[0][1].set_xticks(np.arange(0, 0.6, 0.1)) 
    a[0][1].set_yticks(yaxes)   
    a[1][0].set_title('Missing Completely At Random')
    a[1][0].set_xlabel(xlabel)
    a[1][0].set_ylabel(ylabel)
    a[1][0].set_xticks(np.arange(0, 0.6, 0.1))
    a[1][0].set_yticks(yaxes)  
    a[1][1].set_title('Missing Not At Random')
    a[1][1].set_xlabel(xlabel)
    a[1][1].set_ylabel(ylabel)
    a[1][1].set_xticks(np.arange(0, 0.6, 0.1)) 
    a[1][1].set_yticks(yaxes)   
    fig.suptitle(name+': '+ str(dataset_list)+' '+test+' '+score_type+' Scores for Each Experiment')
    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(savepath + name+'_'+ str(dataset_list)+'_'+test+'_'+score_type+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return

for test in ['Train', 'Test']:
    for scoring in ['rmse', 'explained_var', 'r2', 'training_duration', 'RMSEAcc']:
        display_scores_over_options(reg_data, test=test, score_type=scoring, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis/')

def display_wilcoxon_results(df, savepath, dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'reg'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name != name+'full']

    fulltemp['ID'] = fulltemp['DatasetID']+fulltemp['Condition']+fulltemp['Level']+fulltemp['Triplicate']
    simpletemp['ID'] = simpletemp['DatasetID']+simpletemp['Condition']+simpletemp['Level']+simpletemp['Triplicate']

    fulltemp = fulltemp[fulltemp.ID.isin(simpletemp.ID.unique().tolist())]
    simpletemp = simpletemp[simpletemp.ID.isin(fulltemp.ID.unique().tolist())]
    
    fulltemp.drop(columns=['ID'])
    simpletemp.drop(columns=['ID'])

    name = 'Regression'
    full_frame = pd.DataFrame()
    for score_type in ['rmse', 'explained_var', 'r2', 'training_duration', 'RMSEAcc']:
        match score_type:
            case 'rmse':
                imputer = 'Exp2impute_rmse'
                s_first = 'Exp2impute_rmse'
                complexer = 'Exp3impute_rmse'
                simpler = 'Exp3impute_rmse'
                ylabel = 'RMSE Score'
            case 'explained_var':
                imputer = 'Exp2impute_explained_var'
                s_first = 'Exp2impute_explained_var'
                complexer = 'Exp3impute_explained_var'
                simpler = 'Exp3impute_explained_var'
                ylabel = 'Explained Variance (%)'
            case 'r2':
                imputer = 'Exp2impute_r2'
                s_first = 'Exp2impute_r2'
                complexer = 'Exp3impute_r2'
                simpler = 'Exp3impute_r2'
                ylabel = r'$R_2$'
            case 'training_duration':
                imputer = 'Exp2duration'
                s_first = 'Exp2duration'
                complexer = 'Exp3duration'
                simpler = 'Exp3duration'
                ylabel = 'Training Time (Seconds)'
            case 'RMSEAcc':
                imputer = 'Exp1ImputeRMSEAcc'
                s_first = 'Exp1ImputeRMSEAcc'
                complexer = 'Exp3ImputeRMSEAcc'
                simpler = 'Exp3ImputeRMSEAcc'
                ylabel = 'Imputation Accurcy (RMSE)'
        

    
        all_models = []
        

        for i, space in enumerate([imputer, complexer, simpler, s_first]):
            if i >= 2:
                all_list = simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
                
            else:
                all_list= fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values
                #print(fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            all_models.append(all_list)
        
        all_out = pd.DataFrame([all_models[0],all_models[1], all_models[2], all_models[3]]).T
        all0 = all_out[0].to_frame(name=score_type)
        all0['Model'] = 'Impute_First'
        all1 = all_out[1].to_frame(name=score_type)
        all1['Model'] = 'Complex'
        all2 = all_out[2].to_frame(name=score_type)
        all2['Model'] = 'Simple'
        all3 = all_out[3].to_frame(name=score_type)
        all3['Model'] = 'Simple_First'
        correct_format = pd.concat([all0, all1, all2, all3])
        full_frame = pd.concat([full_frame,correct_format], axis=1)
    full_frame = full_frame.loc[:,~full_frame.columns.duplicated()].copy()
    full_frame.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg_kw_test.csv')

all_out=display_wilcoxon_results(reg_data, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis')
   


    