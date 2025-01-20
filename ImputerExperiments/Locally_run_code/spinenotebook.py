import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy
import re
from numpy.random import RandomState
from permute.core import one_sample

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = '/common/ketrong/tpotexp/tpot2/ImputerExperiments/data/c/spine_first.csv'

class_data = pd.read_csv(path)

class_data.head(5)

#class_data = class_data.dropna(how='any')

class_data.head(5)

class_data = class_data.replace('_', '', regex=True)
class_data = class_data.replace('/', '', regex=True)
class_data.head(5)
class_data.drop(columns=class_data.columns[0], axis=1, inplace=True)
convert_dict = {'DatasetID': str}
class_data = class_data.astype(convert_dict)
class_data.head(5)

class_data = class_data.sort_values(by=['DatasetID', 'Triplicate'], ascending=True)
print(class_data.head(-1))

print('f1_macro')
print(class_data['Exp3impute_f1'].mean())
print(class_data['Exp2impute_f1'].mean())
print(class_data['Exp3impute_f1'].mean()-class_data['Exp2impute_f1'].mean())
prng = RandomState(42)
(p, diff_means) = one_sample(class_data['Exp3impute_f1']-class_data['Exp2impute_f1'],alternative='two-sided', stat='mean', seed=prng)
print("P-value: ", p)
print("Difference in means:", diff_means)

print('log loss')
prng = RandomState(42)
(p, diff_means) = one_sample(class_data['Exp3impute_logloss']-class_data['Exp2impute_logloss'],alternative='two-sided', stat='mean', seed=prng)
print("P-value: ", p)
print("Difference in means:", diff_means)

print('balanced accuracy')
prng = RandomState(42)
(p, diff_means) = one_sample(class_data['Exp3impute_balanced_accuracy']-class_data['Exp2impute_balanced_accuracy'],alternative='two-sided', stat='mean', seed=prng)
print("P-value: ", p)
print("Difference in means:", diff_means)

print('accuracy')
prng = RandomState(42)
(p, diff_means) = one_sample(class_data['Exp3impute_accuracy']-class_data['Exp2impute_accuracy'],alternative='two-sided', stat='mean', seed=prng)
print("P-value: ", p)
print("Difference in means:", diff_means)

print('auroc')
prng = RandomState(42)
(p, diff_means) = one_sample(class_data['Exp3impute_auroc']-class_data['Exp2impute_auroc'], alternative='two-sided', stat='mean', seed=prng)
print("P-value: ", p)
print("Difference in means:", diff_means)