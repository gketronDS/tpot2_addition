import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy
import re

pd.set_option('display.max_columns', None)

path = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/class.csv'

class_data = pd.read_csv(path)

class_data.head(5)

class_data = class_data.dropna(how='any')

class_data.head(5)

class_data = class_data.replace('_', '', regex=True)
class_data = class_data.replace('/', '', regex=True)
class_data.head(5)
class_data.drop(columns=class_data.columns[0], axis=1, inplace=True)
convert_dict = {'DatasetID': int}
class_data = class_data.astype(convert_dict)
convert_dict = {'DatasetID': str}
class_data = class_data.astype(convert_dict)
class_data.head(5)

class_data = class_data.sort_values(by=['DatasetID', 'Condition', 'Level', 'Triplicate'], ascending=True)
class_data.head(-1)

#print(class_data[(class_data.Exp_Name == 'classfull') & (class_data.Level == '0.01')]['Exp2ImputeModel'].value_counts())

def display_model_proportions(df, exp, savepath, complex = False, dataset_list=None, show=False):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    if complex:
        name = 'classfull'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Complex'
        
    else:
        name = 'classsimple'
        temp = temp[temp.Exp_Name == name]
        subtitle = 'Simple'
        

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
           subtitle = 'Impute First'
        case 2:
            pipe = 'Exp2ClassifierModel'
            title = 'Classifier Models'
            subtitle = 'Impute First'
        case 3:
            pipe = 'Exp3ImputeModel'
            title = subtitle+' TPOT2 Imputer Models'
        case 4: 
            pipe = 'Exp3ClassifierModel'
            title = subtitle+' TPOT2 Classifier Models'

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
    fig.suptitle('Classification '+subtitle+' Model Space: '+ str(dataset_list)+' Selection Frequency of ' + title)
    lgd=fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    #fig.savefig(savepath + name+'_'+ str(dataset_list)+'_'+pipe+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    all_models['Missing Fraction'] = xvals
    mar_models['Missing Fraction'] = xvals
    mcar_models['Missing Fraction'] = xvals
    mnar_models['Missing Fraction'] = xvals
    all_table = pd.DataFrame(all_models)
    mar_table = pd.DataFrame(mar_models)
    mcar_table = pd.DataFrame(mcar_models)
    mnar_table = pd.DataFrame(mnar_models)
    return all_table, mar_table, mcar_models, mnar_models

for i in range(1,5):
    complexed = True
    all_table, mar_table, mcar_models, mnar_models = display_model_proportions(class_data, exp=i, complex=complexed, savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/')
    all_table= all_table.map('{:.1%}'.format)
    print(all_table)
    if complexed:
        name = 'classfull'
        #temp = temp[temp.Exp_Name == name]
        subtitle = 'Complex'
    else:
        name = 'classsimple'
        #temp = temp[temp.Exp_Name == name]
        subtitle = 'Simple'

    match i:
            case 1: 
                pipe = 'Exp2ImputeModel'
                title = 'Imputer_Models'
                subtitle = 'Impute_First'
            case 2:
                pipe = 'Exp2ClassifierModel'
                title = 'Classifier_Models'
                subtitle = 'Impute_First'
            case 3:
                pipe = 'Exp3ImputeModel'
                title = subtitle+'_TPOT2_Imputer_Models'
            case 4: 
                pipe = 'Exp3ClassifierModel'
                title = subtitle+'_TPOT2_Classifier_Models'
    out = all_table.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/'+name+pipe+title+subtitle+str(i)+'.csv')

def display_scores_over_options(df, test, score_type, savepath,
                                dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'class'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name != name+'full']
    name = 'Classification'
    
    match test:
        case 'Train':
            match score_type:
                case 'f1':
                    imputer = 'Exp2train_f1'
                    complexer = 'Exp3train_f1'
                    simpler = 'Exp3train_f1'
                    ylabel = 'Macro f1 Score (%) '
                case 'auroc':
                    imputer = 'Exp2train_auroc'
                    complexer = 'Exp3train_auroc'
                    simpler = 'Exp3train_auroc'
                    ylabel = 'AUROC (%) '
                case 'accuracy':
                    imputer = 'Exp2train_accuracy'
                    complexer = 'Exp3train_accuracy'
                    simpler = 'Exp3train_accuracy'
                    ylabel = 'Accuracy (%) '
                case 'balanced_accuracy':
                    imputer = 'Exp2train_balanced_accuracy'
                    complexer = 'Exp3train_balanced_accuracy'
                    simpler = 'Exp3train_balanced_accuracy'
                    ylabel = 'Balanced Accuracy (%)'
                case 'logloss':
                    imputer = 'Exp2train_logloss'
                    complexer = 'Exp3train_logloss'
                    simpler = 'Exp3train_logloss'
                    ylabel = 'Log Loss (%)'
                case 'training_duration':
                    imputer = 'Exp2duration'
                    complexer = 'Exp3duration'
                    simpler = 'Exp3duration'
                    ylabel = 'Training Time (Seconds)'
                case 'RMSEAcc':
                    imputer = 'Exp1ImputeRMSEAcc'
                    complexer = 'Exp3ImputeRMSEAcc'
                    simpler = 'Exp3ImputeRMSEAcc'
                    ylabel = 'Imputation Accuracy (RMSE)'
        case 'Test':
            match score_type:
                case 'f1':
                    imputer = 'Exp2impute_f1'
                    complexer = 'Exp3impute_f1'
                    simpler = 'Exp3impute_f1'
                    ylabel = 'Macro f1 Score (%) '
                case 'auroc':
                    imputer = 'Exp2impute_auroc'
                    complexer = 'Exp3impute_auroc'
                    simpler = 'Exp3impute_auroc'
                    ylabel = 'AUROC (%) '
                case 'accuracy':
                    imputer = 'Exp2impute_accuracy'
                    complexer = 'Exp3impute_accuracy'
                    simpler = 'Exp3impute_accuracy'
                    ylabel = 'Accuracy (%) '
                case 'balanced_accuracy':
                    imputer = 'Exp2impute_balanced_accuracy'
                    complexer = 'Exp3impute_balanced_accuracy'
                    simpler = 'Exp3impute_balanced_accuracy'
                    ylabel = 'Balanced Accuracy (%)'
                case 'logloss':
                    imputer = 'Exp2impute_logloss'
                    complexer = 'Exp3impute_logloss'
                    simpler = 'Exp3impute_logloss'
                    ylabel = 'Log Loss (%)'
                case 'training_duration':
                    imputer = 'Exp2inference_duration'
                    complexer = 'Exp3inference_duration'
                    simpler = 'Exp3inference_duration'
                    ylabel = 'Inference Time (Seconds)'
                case 'RMSEAcc':
                    imputer = 'Exp1ImputeRMSEAcc'
                    complexer = 'Exp3ImputeRMSEAcc'
                    simpler = 'Exp3ImputeRMSEAcc'
                    ylabel = 'Imputation Accurcy (RMSE)'


    xvals = [0.01, 0.1, 0.3, 0.5]
    xlabel = 'Percent Missing'
    all_models = {}
    mar_models = {}
    mcar_models = {}
    mnar_models = {}


    for i, model in enumerate([imputer, complexer, simpler]):
        all_list = []
        for val in xvals:
            if i == 2:
                try:
                    all_list.append(simpletemp[simpletemp.Level == str(val)][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[fulltemp.Level == str(val)][model].mean())
                except:
                    all_list.append(0.0)
        if i == 2:
            all_models['simple_'+model] = all_list
        else:
            all_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler]):
        all_list = []
        for val in xvals:
            if i == 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i == 2:
            mar_models['simple_'+model] = all_list
        else:
            mar_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler]):
        all_list = []
        for val in xvals:
            if i == 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MCAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i == 2:           
            mcar_models['simple_'+model] = all_list
        else:
            mcar_models[model] = all_list
    
    for i, model in enumerate([imputer, complexer, simpler]):
        all_list = []
        for val in xvals:
            if i == 2:
                try:
                    all_list.append(simpletemp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
            else:
                try:
                    all_list.append(fulltemp[(temp.Condition == 'MNAR')&(temp.Level == str(val))][model].mean())
                except:
                    all_list.append(0.0)
        if i == 2:
            mnar_models['simple_'+model] = all_list
        else:
            mnar_models[model] = all_list
    for sets in [all_models, mar_models, mcar_models, mnar_models]:
        sets['Impute First '+score_type] = sets[imputer]
        sets['Complex '+score_type] = sets[complexer]
        sets['Simple '+score_type] = sets['simple_'+simpler]
        del sets[imputer], sets[complexer], sets['simple_'+simpler]
    
    fig, a = plt.subplots(2,2,sharey=True)
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

    yaxes = np.arange(0, max(maxed), 0.2)
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
    fig.savefig(savepath + name+'_'+ str(dataset_list)+'_'+score_type+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return

display_scores_over_options(class_data, test='Test', score_type='f1', savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/')

def rainplot_annotate_brackets(num1, num2, data, size, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=3):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data.pvalue) is str:
        text = data.pvalue
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data.pvalue < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'
        
        if data.pvalue < 0.0001:
            text = text + '\n p=' + re.sub("[$@.&?].*[$@e&?]", "", str(data.pvalue))[:1] + 'e' + re.sub("[$@.&?].*[$@e&?]", "", str(data.pvalue))[1:]+'\n Effect=' + str(round(np.abs(data.zstatistic/np.sqrt(size)),2))

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_xlim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    bary = [lx, lx, rx, rx]
    barx = [y, y+barh, y+barh, y]
    mid = (y+barh, (lx+rx)*0.97/2)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom', backgroundcolor='white', alpha=1.0)
    if fs is not None:
        kwargs['fontsize'] = fs
    
    

    plt.text(*mid, text, **kwargs)

import scipy.stats

def wilcoxon_rainplot(data_x, scorer='Values', title='RaincloudPlot'):

    fig, ax = plt.subplots(figsize=(8, 4))

    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ['yellowgreen', 'olivedrab', 'purple']

    # Boxplot data
    bp = ax.boxplot(data_x, patch_artist = True, vert = False)

    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    for line in bp['medians']:
        # get position data for median line
        x, y = line.get_xydata()[0] # top of median line
        # overlay median value
        plt.text(x, y, '%.3f' % x, horizontalalignment='center') # draw above, centered

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['thistle', 'orchid', 'red']

    # Violinplot data
    vp = ax.violinplot(data_x, points=500, 
                showmeans=False, showextrema=False, showmedians=False, vert=False)

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
        # Change to the desired color
        b.set_color(violin_colors[idx])



    heights = [max(i) for i in data_x]
    x1x2 = scipy.stats.wilcoxon(x1, x2, method='approx')
    x2x3 = scipy.stats.wilcoxon(x2, x3, method='approx')
    x1x3 =scipy.stats.wilcoxon(x1, x3, method='approx')
    print(x1x3)
    bars = np.arange(1,4,1)
    rainplot_annotate_brackets(0, 1, x1x2, x1.shape[0], bars, heights)
    rainplot_annotate_brackets(1, 2, x2x3, x2.shape[0], bars, heights)
    rainplot_annotate_brackets(0, 2, x1x3, x1.shape[0], bars, heights, dh=.2)


    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ['tomato', 'darksalmon', 'teal']


    # Scatterplot data
    for idx, features in enumerate(data_x):
    
        bins = np.arange(min(features), max(features), step=(max(features)-min(features))/11)
        hist, edges = np.histogram(features, bins=bins)
        #print(len(hist))
        scalehist = -0.3*(hist-hist.min())/(hist.max()-hist.min())
        #print(len(scalehist))
        #print(edges)
        #y = np.full(len(bins), idx + .8)
        #idxs=np.arange(1, hist.max()+1)


        y = np.arange(idx+0.8, scalehist.min()+idx+0.8, step=scalehist.min()/10)
        x = np.arange(min(features)+(max(features)-min(features))/20,max(features)+(max(features)-min(features))/20, step=(max(features)-min(features))/10)
        X,Y = np.meshgrid(x,y)
        #print(scalehist)
        Y[Y<scalehist+idx+0.8] = np.nan
        
        plt.scatter(X,Y, s=1, c=scatter_colors[idx])
        
        

        
        '''
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        plt.scatter(features, y, s=.3, c=scatter_colors[idx])
        '''



    plt.yticks(np.arange(1,4,1), ['Impute First', 'Complex', 'Simple'])  # Set text labels.
    plt.xlabel(scorer)
    plt.ylabel('Experiments')
    plt.title(title)
    plt.show()

def display_wilcoxon_results(df, score_type, savepath, dataset_list=None):
    if dataset_list is not None:
        temp = df.loc[df['DatasetID'].isin(dataset_list)].copy()
    else:
        temp = df.copy()
        dataset_list = 'All Datasets'
    
    #select temp datasets for impute first, complex, and simple to compare across model settings
    
    name = 'class'
    fulltemp = temp[temp.Exp_Name == name+'full']
    simpletemp = temp[temp.Exp_Name != name+'full']

    fulltemp['ID'] = fulltemp['DatasetID']+fulltemp['Condition']+fulltemp['Level']+fulltemp['Triplicate']
    simpletemp['ID'] = simpletemp['DatasetID']+simpletemp['Condition']+simpletemp['Level']+simpletemp['Triplicate']

    fulltemp = fulltemp[fulltemp.ID.isin(simpletemp.ID.unique().tolist())]
    simpletemp = simpletemp[simpletemp.ID.isin(fulltemp.ID.unique().tolist())]
    
    fulltemp.drop(columns=['ID'])
    simpletemp.drop(columns=['ID'])

    name = 'Classification'
    full_frame = pd.DataFrame()
    for score_type in ['f1', 'auroc', 'accuracy', 'balanced_accuracy', 'logloss', 'training_duration', 'RMSEAcc']:
        match score_type:
            case 'f1':
                imputer = 'Exp2impute_f1'
                complexer = 'Exp3impute_f1'
                simpler = 'Exp3impute_f1'
                ylabel = 'Weighted f1 Score (%) '
            case 'auroc':
                imputer = 'Exp2impute_auroc'
                complexer = 'Exp3impute_auroc'
                simpler = 'Exp3impute_auroc'
                ylabel = 'AUROC (%) '
            case 'accuracy':
                imputer = 'Exp2impute_accuracy'
                complexer = 'Exp3impute_accuracy'
                simpler = 'Exp3impute_accuracy'
                ylabel = 'Accuracy (%) '
            case 'balanced_accuracy':
                imputer = 'Exp2impute_balanced_accuracy'
                complexer = 'Exp3impute_balanced_accuracy'
                simpler = 'Exp3impute_balanced_accuracy'
                ylabel = 'Balanced Accuracy (%)'
            case 'logloss':
                imputer = 'Exp2impute_logloss'
                complexer = 'Exp3impute_logloss'
                simpler = 'Exp3impute_logloss'
                ylabel = 'Log Loss'
            case 'training_duration':
                imputer = 'Exp2duration'
                complexer = 'Exp3duration'
                simpler = 'Exp3duration'
                ylabel = 'Training Time (Seconds)'
            case 'RMSEAcc':
                imputer = 'Exp1ImputeRMSEAcc'
                complexer = 'Exp3ImputeRMSEAcc'
                simpler = 'Exp3ImputeRMSEAcc'
                ylabel = 'Imputation Accurcy (RMSE)'
            
        all_models = []
        

        for i, space in enumerate([imputer, complexer, simpler]):
            if i == 2:
                all_list = simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values
                #print(simpletemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
                
            else:
                all_list= fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values
                #print(fulltemp.sort_values(by=['DatasetID','Condition', 'Level', 'Triplicate'], ascending=True)[space].values)
            all_models.append(all_list)
        
        all_out = pd.DataFrame([all_models[0],all_models[1], all_models[2]]).T
        all0 = all_out[0].to_frame(name=score_type)
        all0['Model'] = 'Impute_F'
        all1 = all_out[1].to_frame(name=score_type)
        all1['Model'] = 'Complex'
        all2 = all_out[2].to_frame(name=score_type)
        all2['Model'] = 'Simple'
        correct_format = pd.concat([all0, all1, all2])
        full_frame = pd.concat([full_frame,correct_format], axis=1)

    full_frame.to_csv('/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/class_kw_test.csv')


display_wilcoxon_results(class_data, score_type='training_duration',savepath='/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/Saved_Analysis/')