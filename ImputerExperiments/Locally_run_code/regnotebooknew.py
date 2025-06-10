import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy
import re
import seaborn as sns
import scipy.stats
sns.set_theme()
pd.set_option('display.max_columns', None)

path = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg_added42545.csv'
#path2 = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg_non_simple.csv'

reg_data = pd.read_csv(path)
#reg_data_non = pd.read_csv(path2)
reg_data.head(5)

reg_data = reg_data.dropna(how='any')
#reg_data_non = reg_data_non.dropna(how='any', axis=0)
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
#path 2
#reg_data_non = reg_data_non.replace('_', '', regex=True)
#reg_data_non = reg_data_non.replace('/', '', regex=True)
#print('replacement')
#print(reg_data_non)
#reg_data_non.drop(columns=reg_data_non.columns[0], axis=1, inplace=True)
#convert_dict = {'DatasetID': int}
#reg_data_non = reg_data_non.astype(convert_dict)
#convert_dict = {'DatasetID': str}
#reg_data_non = reg_data_non.astype(convert_dict)

#reg_data_non = reg_data_non.sort_values(by=['DatasetID', 'Condition', 'Level', 'Triplicate'], ascending=True)
reg_data_new = reg_data #pd.concat([reg_data, reg_data_non], axis=0)
#print(class_data[(class_data.Exp_Name == 'classfull') & (class_data.Level == '0.01')]['Exp2ImputeModel'].value_counts())

def display_model_proportions(df, exp, savepath, type, dataset_list=None, show=False):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    if type == 'complex':
        name = 'regfull'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Complex'
        sub2 = "Impute First"
        
    if type == 'simple':
        name = 'regsimple'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Simple'
        sub2 = 'Simple First'
    
    if type == 'non_simple':
        name = 'regnonsimple'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Non_simple'
        sub2 = "No Imputation"
        

    xvals = [0.01, 0.1, 0.3, 0.5]
    xlabel = 'Percent Missing (%)'
    ylabel = 'Percent of Time Selected (%)'

    all_models = {}
    #mar_models = {}
    #mcar_models = {}
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
    '''
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
        mcar_models[model] = new_list3'
    '''
    for model in temp[temp.Condition == 'MNAR'][pipe].unique():
        new_list4 = []
        for val in xvals:
            try:
                new_list4.append(temp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][pipe].value_counts()[model]/temp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][pipe].value_counts().sum())
            except:
                new_list4.append(0.0)
        mnar_models[model] = new_list4
    for i, label in enumerate(all_models):
        plt.plot(xvals, mnar_models[label], color="C"+str(i))
        plt.title('Missing Not At Random')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(np.arange(0, 0.6, 0.1))  
        plt.yticks(np.arange(0, 1.1, 0.2)) 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.show()
    '''
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
    mcar_models['Missing Fraction'] = xvals'
    '''
    mnar_models['Missing_Fraction'] = xvals
    #all_table = pd.DataFrame(all_models)
    #mar_table = pd.DataFrame(mar_models)
    #mcar_table = pd.DataFrame(mcar_models)
    mnar_table = pd.DataFrame(mnar_models)
    return mnar_table, mnar_models

for twos in ['complex', 'simple', 'non_simple',]:
    for i in range(1,5):
        complexed = twos
        mnar_table, mnar_models = display_model_proportions(reg_data_new, exp=i, type=complexed, savepath='/Users/gabrielketron/tpot2_addimputers/tpot2/ImputerExperiments/data/r/Saved_Analysis/')
        #all_table= all_table.map('{:.0%}'.format)
        #print(mnar_table)
        if twos == 'complex':
            name = 'regfull'
            #temp = temp[temp.Exp_Name == name]
            subtitle = 'Complex'
            sub2 = "Impute First"
        
        elif twos == 'simple':
            name = 'regsimple'
            #temp = temp[temp.Exp_Name == name]
            subtitle = 'Simple'
            sub2 = 'Simple First'
        
        elif twos == 'non_simple':
            name = 'regnonsimple'
            #temp = temp[temp.Exp_Name == name]
            subtitle = 'Non_simple'
            sub2 = "No Imputation"

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
        mnar_table = mnar_table.T
        mnar_table['Total'] = mnar_table.mean(axis=1)
        mnar_table= mnar_table.map('{:.0%}'.format)
        mnar_table.columns = mnar_table.iloc[-1]
        mnar_table = mnar_table.add_suffix(" Missing")
        mnar_table.columns = [*mnar_table.columns[:-1], 'Total']
        mnar_table = mnar_table.drop('Missing_Fraction')
        out = mnar_table.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis/'+name+pipe+title+subtitle+str(i)+'mnar_non_only.csv')

def mean_confidence_interval(data):
    a = 1.0 * np.array(data)
    n = len(a)
    #print(n)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * 1.96
    return m, -h, h

def std_interval(data):
    a = 1.0 * np.array(data)
    n = len(a)
    #print(n)
    m = np.mean(a)
    h = np.std(a)
    return m, h*-1, h

def display_scores_over_options(df, score_type, savepath,
                                dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'reg'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name == name+'simple']
    nontemp = temp[temp.Exp_Name == name+'nonsimple']
    name = 'Regression'

    
    match score_type:
        case 'rmse':
            titlename = 'Prediction RMSE'
            imputer = 'Exp2train_rmse'
            s_first = 'Exp2train_rmse'
            no_imp ='Exp2train_rmse'
            complexer = 'Exp3train_rmse'
            simpler = 'Exp3train_rmse'
            non_simpler ='Exp3train_rmse'
            ylabel = 'RMSE Score'
            timputer = 'Exp2impute_rmse'
            ts_first = 'Exp2impute_rmse'
            tno_imp ='Exp2impute_rmse'
            tcomplexer = 'Exp3impute_rmse'
            tsimpler = 'Exp3impute_rmse'
            tnon_simpler = 'Exp3impute_rmse'
            tylabel = 'RMSE Score'
        case 'explained_var':
            titlename = 'Explained Variance'
            imputer = 'Exp2train_explained_var'
            s_first = 'Exp2train_explained_var'
            no_imp ='Exp2train_explained_var'
            complexer = 'Exp3train_explained_var'
            simpler = 'Exp3train_explained_var'
            non_simpler ='Exp3train_explained_var'
            ylabel = 'Explained Variance (%)'
            timputer = 'Exp2impute_explained_var'
            ts_first = 'Exp2impute_explained_var'
            tno_imp ='Exp2impute_explained_var'
            tcomplexer = 'Exp3impute_explained_var'
            tsimpler = 'Exp3impute_explained_var'
            tnon_simpler = 'Exp3impute_explained_var'
            tylabel = 'Explained Variance (%)'
        case 'r2':
            titlename = r'$R_2$'
            imputer = 'Exp2train_r2'
            s_first = 'Exp2train_r2'
            no_imp ='Exp2train_r2'
            complexer = 'Exp3train_r2'
            simpler = 'Exp3train_r2'
            non_simpler ='Exp3train_r2'
            ylabel = r'$R_2$'
            timputer = 'Exp2impute_r2'
            ts_first = 'Exp2impute_r2'
            tno_imp ='Exp2impute_r2'
            tcomplexer = 'Exp3impute_r2'
            tsimpler = 'Exp3impute_r2'
            tnon_simpler = 'Exp3impute_r2'
            tylabel = r'$R_2$'
        case 'training_duration':
            titlename = 'Training Duration'
            imputer = 'Exp2duration'
            s_first = 'Exp2duration'
            no_imp ='Exp2duration'
            complexer = 'Exp3duration'
            simpler = 'Exp3duration'
            non_simpler ='Exp3duration'
            ylabel = 'Training Time (Seconds)'
            timputer = 'Exp2inference_duration'
            ts_first = 'Exp2inference_duration'
            tno_imp ='Exp2inference_duration'
            tcomplexer = 'Exp3inference_duration'
            tsimpler = 'Exp3inference_duration'
            tnon_simpler = 'Exp3inference_duration'
            tylabel = 'Inference Time (Seconds)'
        case 'RMSEAcc':
            titlename = 'Imputation Quality RMSE'
            imputer = 'Exp1ImputeRMSEAcc'
            s_first = 'Exp1ImputeRMSEAcc'
            no_imp ='Exp1ImputeRMSEAcc'
            complexer = 'Exp3TrainRMSEAcc'
            simpler = 'Exp3TrainRMSEAcc'
            non_simpler = 'Exp3TrainRMSEAcc'
            ylabel = 'Imputation RMSE'    
            timputer = 'Exp1ImputeRMSEAcc'
            ts_first = 'Exp1ImputeRMSEAcc'
            tno_imp ='Exp1ImputeRMSEAcc'
            tcomplexer = 'Exp3ImputeRMSEAcc'
            tsimpler = 'Exp3ImputeRMSEAcc'
            tnon_simpler = 'Exp3ImputeRMSEAcc'
            tylabel = 'Imputation RMSE'     
       

    xvals = [0.01, 0.1, 0.3, 0.5]
    xlabel = 'Percent Missing'
    train_mnar_models = {}
    test_mnar_models = {}
    train_error = {}
    test_error = {}


    for i, model in enumerate([complexer, simpler, non_simpler, no_imp]):
        all_list = []
        cilow =[]
        cihigh = []
        for val in xvals:
            if i == 1:
                try:
                    m, mlow, mhigh= mean_confidence_interval(simpletemp[(simpletemp.Condition == 'MNAR')&(simpletemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
            elif i == 2 or i == 3:
                try:
                    m, mlow, mhigh= mean_confidence_interval(nontemp[(nontemp.Condition == 'MNAR')&(nontemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
            else:
                try:
                    m, mlow, mhigh= mean_confidence_interval(fulltemp[(fulltemp.Condition == 'MNAR')&(fulltemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
            #print(i, val, m, mlow, mhigh)
        if i == 1:
            #print("simple")
            #print(all_list)
            train_mnar_models['simple_'+model] = all_list
            train_error['simple_'+model] = cihigh
        elif i == 2 or i == 3:
            #print("nonsimple or noimp")
            #print(all_list)
            train_mnar_models['non_'+model] = all_list
            train_error['non_'+model] = cihigh
        else:
            #print("mixed")
            #print(all_list)
            train_mnar_models[model] = all_list
            train_error[model] = cihigh
    
    #all_models, mar_models, mcar_models, 
    for sets in [train_mnar_models]:
        #sets['Impute First '+score_type] = sets[imputer]
        sets['Mixed '+titlename] = sets[complexer]
        sets['Simple '+titlename] = sets['simple_'+simpler]
        #sets['Simple First '+score_type] = sets['simple_'+s_first]
        sets['Non-Simple '+titlename] = sets['non_'+non_simpler]
        sets['No Imputation '+titlename] = sets['non_'+no_imp]
        del sets[complexer], sets['simple_'+simpler], sets['non_'+non_simpler], sets['non_'+no_imp]
    
    for sets in [train_error]:
        #sets['Impute First '+score_type] = sets[imputer]
        sets['Mixed '+titlename] = sets[complexer]
        sets['Simple '+titlename] = sets['simple_'+simpler]
        #sets['Simple First '+score_type] = sets['simple_'+s_first]
        sets['Non-Simple '+titlename] = sets['non_'+non_simpler]
        sets['No Imputation '+titlename] = sets['non_'+no_imp]
        del sets[complexer], sets['simple_'+simpler], sets['non_'+non_simpler], sets['non_'+no_imp]

    for i, model in enumerate([tcomplexer, tsimpler, tnon_simpler, tno_imp]):
        all_list = []
        cilow =[]
        cihigh = []
        for val in xvals:
            if i == 1:
                try:
                    m, mlow, mhigh= mean_confidence_interval(simpletemp[(simpletemp.Condition == 'MNAR')&(simpletemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
            elif i == 2 or i == 3:
                try:
                    m, mlow, mhigh= mean_confidence_interval(nontemp[(nontemp.Condition == 'MNAR')&(nontemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
            else:
                try:
                    m, mlow, mhigh= mean_confidence_interval(fulltemp[(fulltemp.Condition == 'MNAR')&(fulltemp.Level == str(val))][model])
                    all_list.append(m)
                    cilow.append(mlow)
                    cihigh.append(mhigh)
                except:
                    all_list.append(0.0)
                    cilow.append(0.0)
                    cihigh.append(0.0)
        if i == 1:
            test_mnar_models['simple_'+model] = all_list
            test_error['simple_'+model] = cihigh

        elif i == 2 or i == 3:
            test_mnar_models['non_'+model] = all_list
            test_error['non_'+model] = cihigh
        else:
            test_mnar_models[model] = all_list
            test_error[model] = cihigh
    
    #all_models, mar_models, mcar_models, 
    for sets in [test_mnar_models]:
        #sets['Impute First '+score_type] = sets[imputer]
        sets['Mixed '+titlename] = sets[tcomplexer]
        sets['Simple '+titlename] = sets['simple_'+tsimpler]
        #sets['Simple First '+score_type] = sets['simple_'+s_first]
        sets['Non-Simple '+titlename] = sets['non_'+tnon_simpler]
        sets['No Imputation '+titlename] = sets['non_'+tno_imp]
        del sets[tcomplexer], sets['simple_'+tsimpler], sets['non_'+tnon_simpler], sets['non_'+tno_imp]
    
    for sets in [test_error]:
        #sets['Impute First '+score_type] = sets[imputer]
        sets['Mixed '+titlename] = sets[tcomplexer]
        sets['Simple '+titlename] = sets['simple_'+tsimpler]
        #sets['Simple First '+score_type] = sets['simple_'+s_first]
        sets['Non-Simple '+titlename] = sets['non_'+tnon_simpler]
        sets['No Imputation '+titlename] = sets['non_'+tno_imp]
        del sets[tcomplexer], sets['simple_'+tsimpler], sets['non_'+tnon_simpler], sets['non_'+tno_imp]
    
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(12,10))
    fig.text(0.5, 0.94, f'Regression {titlename}', transform=fig.transFigure, fontsize=16, ha='center')
    #print(train_mnar_models)
    #print(xvals)
    maxed = [0]
    for i, label in enumerate(train_mnar_models):
        ax1.plot(xvals,train_mnar_models[label], color="C"+str(i), label=str(label))
        ax1.errorbar(xvals,train_mnar_models[label], yerr = train_error[label], fmt ='o')
        maxed.append(max(train_error[label])+max(train_mnar_models[label]))
        try:
            ax2.plot(xvals, test_mnar_models[label], color="C"+str(i))
            ax2.errorbar(xvals,test_mnar_models[label], yerr = test_error[label], fmt ='o')
            maxed.append(max(test_error[label])+max(test_mnar_models[label]))
        except:
            save = i

    match score_type:
        case 'rmse':
            yaxes = np.arange(0, np.round(6000)+np.round(6000/6, decimals=-3), np.round(6000)/6)
        case 'explained_var':
            yaxes = np.arange(0, 1.2, 0.2)
        case 'r2':
            yaxes = np.arange(0, 1.2, 0.2)
        case 'training_duration':
            yaxes = np.arange(0, np.round(15000)+np.round(15000)/3, np.round(15000)/3)
        case 'RMSEAcc':
            yaxes = np.arange(0, np.round(0.35, decimals=2)+np.round(0.35, decimals=2)/7, np.round(0.35, decimals=2)/7)
    
    #yaxes = np.arange(0, 8+0.5, 1)
    ax1.set_title('Train MNAR')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xticks(np.arange(0, 0.6, 0.1)) 
    ax1.set_yticks(yaxes)       
    ax2.set_title('Test MNAR')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(tylabel)
    ax2.set_xticks(np.arange(0, 0.6, 0.1)) 
    ax2.set_yticks(yaxes)   
    fig.suptitle(name+': '+ str(dataset_list)+' '+score_type+' Scores for Each Experiment')
    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(savepath + name+'_'+ str(dataset_list)+'_'+score_type+'no_imp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return
    

for scoring in ['rmse']:#, 'explained_var', 'r2', 'training_duration', 'RMSEAcc']:
    display_scores_over_options(reg_data_new, score_type=scoring, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis/')

def display_wilcoxon_results(df, savepath, dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'reg'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name == name+'simple']
    nontemp = temp[temp.Exp_Name == name+'nonsimple']

    fulltemp['ID'] = fulltemp['DatasetID']+fulltemp['Condition']+fulltemp['Level']+fulltemp['Triplicate']
    simpletemp['ID'] = simpletemp['DatasetID']+simpletemp['Condition']+simpletemp['Level']+simpletemp['Triplicate']
    nontemp['ID'] = nontemp['DatasetID']+nontemp['Condition']+nontemp['Level']+nontemp['Triplicate']
    
    nontemp = nontemp[nontemp.ID.isin(simpletemp.ID) & nontemp.ID.isin(fulltemp.ID)]
    simpletemp = simpletemp[simpletemp.ID.isin(nontemp.ID)]
    fulltemp = fulltemp[fulltemp.ID.isin(nontemp.ID)]
    #print(fulltemp)
    nontemp = nontemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)
    simpletemp = simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)
    fulltemp = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)

   #print(fulltemp)
    #fulltemp.drop(columns=['ID'])
    #simpletemp.drop(columns=['ID'])
    #nontemp.drop(columns=['ID'])

    name = 'Regression'
    full_frame = pd.DataFrame()
    #full_frame['ID'] = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID']
    for score_type in ['rmse', 'explained_var', 'r2', 'training_duration', 'RMSEAcc']:
        match score_type:
            case 'rmse':
                imputer = 'Exp2impute_rmse'
                s_first = 'Exp2impute_rmse'
                no_imp = 'Exp2impute_rmse'
                complexer = 'Exp3impute_rmse'
                simpler = 'Exp3impute_rmse'
                non_simpler ='Exp3impute_rmse'
                ylabel = 'RMSE Score'
            case 'explained_var':
                imputer = 'Exp2impute_explained_var'
                s_first = 'Exp2impute_explained_var'
                no_imp = 'Exp2impute_explained_var'
                complexer = 'Exp3impute_explained_var'
                simpler = 'Exp3impute_explained_var'
                non_simpler ='Exp3impute_explained_var'
                ylabel = 'Explained Variance (%)'
            case 'r2':
                imputer = 'Exp2impute_r2'
                s_first = 'Exp2impute_r2'
                no_imp = 'Exp2impute_r2'
                complexer = 'Exp3impute_r2'
                simpler = 'Exp3impute_r2'
                non_simpler ='Exp3impute_r2'
                ylabel = r'$R_2$'
            case 'training_duration':
                imputer = 'Exp2duration'
                s_first = 'Exp2duration'
                no_imp = 'Exp2duration'
                complexer = 'Exp3duration'
                simpler = 'Exp3duration'
                non_simpler ='Exp3duration'
                ylabel = 'Training Time (Seconds)'
            case 'RMSEAcc':
                imputer = 'Exp1ImputeRMSEAcc'
                s_first = 'Exp1ImputeRMSEAcc'
                no_imp = 'Exp1ImputeRMSEAcc'
                complexer = 'Exp3ImputeRMSEAcc'
                simpler = 'Exp3ImputeRMSEAcc'
                non_simpler ='Exp3ImputeRMSEAcc'
                ylabel = 'Imputation Accurcy (RMSE)'
        

    
        all_models = []
        dataset=[]

        for i, space in enumerate([imputer, complexer, simpler, s_first, non_simpler, no_imp]):
            if i == 2 or i == 3:
                all_list = simpletemp[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            if i == 4 or i == 5:
                all_list = nontemp[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            if i == 0 or i == 1:
                all_list= fulltemp[space].values
                #print(fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            all_models.append(all_list)
        
        all_out = pd.DataFrame([all_models[0],all_models[1], all_models[2], all_models[3], all_models[4], all_models[5]]).T
        all0 = all_out[0].to_frame(name=score_type)
        all0['Model'] = 'Impute_First'
        all0['ID'] = fulltemp['DatasetID']
        all1 = all_out[1].to_frame(name=score_type)
        all1['Model'] = 'Complex'
        all1['ID'] = fulltemp['DatasetID']
        all2 = all_out[2].to_frame(name=score_type)
        all2['Model'] = 'Simple'
        all2['ID'] = simpletemp['DatasetID']
        all3 = all_out[3].to_frame(name=score_type)
        all3['Model'] = 'Simple_First'
        all3['ID'] = simpletemp['DatasetID']
        all4 = all_out[4].to_frame(name=score_type)
        all4['Model'] = 'NonSimple'
        all4['ID'] = nontemp['DatasetID']
        all5 = all_out[5].to_frame(name=score_type)
        all5['Model'] = 'NoImpute'
        all5['ID'] = nontemp['DatasetID']
        correct_format = pd.concat([all0, all1, all2, all3, all4, all5])
        print(correct_format)
        #correct_format = pd.concat([fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID'], correct_format])
        full_frame = pd.concat([full_frame,correct_format], axis=1)
    full_frame = full_frame.loc[:,~full_frame.columns.duplicated()].copy()
    #full_frame['ID'] = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID']
    full_frame.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg_kw_test_mnar_fixed.csv')

all_out=display_wilcoxon_results(reg_data_new, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis')
   
def display_wilcoxon_results2(df, savepath, dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'reg'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name == name+'simple']
    nontemp = temp[temp.Exp_Name == name+'nonsimple']

    fulltemp['ID'] = fulltemp['DatasetID']+fulltemp['Condition']+fulltemp['Level']+fulltemp['Triplicate']
    simpletemp['ID'] = simpletemp['DatasetID']+simpletemp['Condition']+simpletemp['Level']+simpletemp['Triplicate']
    nontemp['ID'] = nontemp['DatasetID']+nontemp['Condition']+nontemp['Level']+nontemp['Triplicate']
    
    nontemp = nontemp[nontemp.ID.isin(simpletemp.ID) & nontemp.ID.isin(fulltemp.ID)]
    simpletemp = simpletemp[simpletemp.ID.isin(nontemp.ID)]
    fulltemp = fulltemp[fulltemp.ID.isin(nontemp.ID)]

    nontemp = nontemp.drop(nontemp[nontemp['Level'] != '0.01'].index)
    simpletemp = simpletemp.drop(simpletemp[simpletemp['Level'] != '0.01'].index)
    fulltemp = fulltemp.drop(fulltemp[fulltemp['Level'] != '0.01'].index)
    #print(fulltemp)
    nontemp = nontemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)
    simpletemp = simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)
    fulltemp = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True, ignore_index=True)

    print(fulltemp)
    #fulltemp.drop(columns=['ID'])
    #simpletemp.drop(columns=['ID'])
    #nontemp.drop(columns=['ID'])

    name = 'Regression'
    full_frame = pd.DataFrame()
    #full_frame['ID'] = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID']
    for j in ['ori', 'impute']:
        match j:
            case 'ori':
                complexer = 'Exp2ori_rmse'
                simpler = 'Exp2ori_rmse'
                non_simpler ='Exp2ori_rmse'
                ylabel = 'RMSE Score'
            case 'impute':
                complexer = 'Exp3impute_rmse'
                simpler = 'Exp3impute_rmse'
                non_simpler ='Exp3impute_rmse'
                ylabel = 'RMSE Score'
            
    
        all_models = []
        dataset=[]

        for i, space in enumerate([complexer, simpler, non_simpler]):
            if i == 1:
                all_list = simpletemp[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            if i == 2:
                all_list = nontemp[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            if i == 0:
                all_list= fulltemp[space].values
                #print(fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            all_models.append(all_list)
        
        all_out = pd.DataFrame([all_models[0],all_models[1], all_models[2]]).T
        all0 = all_out[0].to_frame(name=j)
        all0['Model'] = 'Mixed'
        all0['ID'] = fulltemp['DatasetID']
        all1 = all_out[1].to_frame(name=j)
        all1['Model'] = 'Simple'
        all1['ID'] = fulltemp['DatasetID']
        all2 = all_out[2].to_frame(name=j)
        all2['Model'] = 'NonSimple'
        all2['ID'] = nontemp['DatasetID']
        
        correct_format = pd.concat([all0, all1, all2])
        print(correct_format)
        #correct_format = pd.concat([fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID'], correct_format])
        full_frame = pd.concat([full_frame,correct_format], axis=1)
    full_frame = full_frame.loc[:,~full_frame.columns.duplicated()].copy()
    #full_frame['ID'] = fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)['ID']
    full_frame.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/reg_wilcox_test_mnar.csv')

all_out=display_wilcoxon_results2(reg_data_new, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/r/Saved_Analysis')
    