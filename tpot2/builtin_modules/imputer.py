# %%
# -*- coding: utf-8 -*-

"""Copyright (c) 2015 The auto-sklearn developers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the auto-sklearn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
#TODO support np arrays

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from sklearn.preprocessing import OneHotEncoder
import sklearn
import sklearn.impute

import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.compose

import torch

class ColumnSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self,  columns="all",         
                        missing_values=np.nan,
                        strategy="mean",
                        fill_value=None,
                        copy=True,
                        add_indicator=False,
                        keep_empty_features=False,):
        
        self.columns = columns
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features


    def fit(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Dense array or sparse matrix.
        y: array-like {n_samples,} (Optional, ignored)
            Feature labels
        """

        if (self.columns == "categorical" or self.columns == "numeric") and not isinstance(X, pd.DataFrame):
            raise ValueError(f"Invalid value for columns: {self.columns}. "
                             "Only 'all' or <list> is supported for np arrays")

        if self.columns == "categorical":
            self.columns_ = list(X.select_dtypes(exclude='number').columns)
        elif self.columns == "numeric":
            self.columns_ =  [col for col in X.columns if is_numeric_dtype(X[col])]
        elif self.columns == "all":
            if isinstance(X, pd.DataFrame):
                self.columns_ = X.columns
            else:
                self.columns_ = list(range(X.shape[1]))
        elif isinstance(self.columns, list):
            self.columns_ = self.columns
        else:
            raise ValueError(f"Invalid value for columns: {self.columns}")
        
        if len(self.columns_) == 0:
            return self
        
        self.imputer = sklearn.impute.SimpleImputer(missing_values=self.missing_values,
                                                    strategy=self.strategy,
                                                    fill_value=self.fill_value,
                                                    copy=self.copy,
                                                    add_indicator=self.add_indicator,
                                                    keep_empty_features=self.keep_empty_features)
        
        if isinstance(X, pd.DataFrame):
            self.imputer.set_output(transform="pandas")

        if isinstance(X, pd.DataFrame):
            self.imputer.fit(X[self.columns_], y)
        else:
            self.imputer.fit(X[:, self.columns_], y)

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Dense array or sparse matrix.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        if len(self.columns_) == 0:
            return X

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X[self.columns_] = self.imputer.transform(X[self.columns_])
            return X
        else:
            X = np.copy(X)
            X[:, self.columns_] = self.imputer.transform(X[:, self.columns_])
            return X

class GainImputer(BaseEstimator, TransformerMixin):
    """
    Base class for all imputers.
    It adds automatically support for `add_indicator`.
    """

    def __init__(self, 
                 batch_size=128, 
                 hint_rate=0.9, 
                 alpha=100, 
                 iterations=10000,
                 train_rate = 0.8,
                 learning_rate = 0.001,
                 p_miss = 0.2,
                 random_state=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.train_rate = train_rate
        self.learning_rate = learning_rate
        self.p_miss = p_miss
        self.random_state = random_state
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.float32)
        torch.set_grad_enabled(True)
        if random_state is not None:
            torch.manual_seed(self.random_state)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X, y = None):
        
        self.modelG.load_state_dict(self._Gen_params)

        if hasattr(X, 'dtypes'):
            X = X.to_numpy()
        #define mask matrix
        X_mask = 1 - np.isnan(X)
        #get dimensions
        no, self.dim = X.shape
        self.int_dim = int(self.dim)
        #normalize the original data, and save parameters for renormalization
        norm_data = X.copy()
        min_val = np.zeros(self.dim)
        max_val = np.zeros(self.dim)
        for i in range(self.dim):
            min_val[i] = np.nanmin(norm_data[i])
            norm_data[:, i] -= np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[i])
            norm_data[:, i] /= (np.nanmax(norm_data[:, i]) + 1e-06)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
        norm_data_filled = np.nan_to_num(norm_data, 0)
        p_miss_vec = self.p_miss * np.ones((self.dim,1)) 
        Missing = np.zeros((no,self.dim))
        for i in range(self.dim):
            A = np.random.uniform(0., 1., size = [len(norm_data_filled),])
            B = A > p_miss_vec[i]
            Missing[:,i] = 1.*B

        Z_mb = self._sample_Z(no, self.dim)
        M_mb = Missing
        X_mb = norm_data_filled

        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        X_mb = torch.tensor(X_mb, dtype=torch.float32)
        New_X_mb = torch.tensor(New_X_mb, dtype=torch.float32)
        M_mb = torch.tensor(M_mb, dtype=torch.float32)

        G_sample = self.modelG(X_mb, New_X_mb, M_mb)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        mse_final = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample)/(1-M_mb).sum()
        print('Final Test RMSE: ' + str(np.sqrt(mse_final.item())))

        imputed_data = M_mb * X_mb + (1-M_mb) * G_sample
        imputed_data = imputed_data.cpu().detach().numpy()
        _, dim = imputed_data.shape
        renorm_data = imputed_data.copy()
        for i in range(dim):
            renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
            renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        for i in range(dim):
            temp = X[~np.isnan(X[:, i]), i]
            # Only for the categorical variable
            if len(np.unique(temp)) < 20:
                renorm_data[:, i] = np.round(renorm_data[:, i])
        return 
        
    def fit_transform(self, X, y=None):
        if hasattr(X, 'dtypes'):
            X = X.to_numpy()
        #define mask matrix
        X_mask = 1 - np.isnan(X)
        #get dimensions
        no, self.dim = X.shape
        self.int_dim = int(self.dim)
        #normalize the original data, and save parameters for renormalization
        norm_data = X.copy()
        min_val = np.zeros(self.dim)
        max_val = np.zeros(self.dim)
        for i in range(self.dim):
            min_val[i] = np.nanmin(norm_data[i])
            norm_data[:, i] -= np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[i])
            norm_data[:, i] /= (np.nanmax(norm_data[:, i]) + 1e-06)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
        norm_data_filled = np.nan_to_num(norm_data, 0)
        p_miss_vec = self.p_miss * np.ones((self.dim,1)) 
        Missing = np.zeros((no,self.dim))
        for i in range(self.dim):
            A = np.random.uniform(0., 1., size = [len(norm_data_filled),])
            B = A > p_miss_vec[i]
            Missing[:,i] = 1.*B
        #internal test-train split
        # Train / Test Missing Indicators
        #model training
        self.modelD = self.Discriminator(GainImputer=self)
        self.modelG = self.Generator(GainImputer=self)

        optimizer_D = torch.optim.Adam(self.modelD.parameters(), 
                                       lr = self.learning_rate)
        optimizer_G = torch.optim.Adam(self.modelG.parameters(), 
                                       lr = self.learning_rate)
        
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        mse_loss = torch.nn.MSELoss(reduction='mean')

        for it in range(self.iterations):
            mb_idx = self._sample_index(no, self.batch_size)
            X_mb = norm_data_filled[mb_idx,:]
            Z_mb = self._sample_Z(self.batch_size, self.dim)

            M_mb = Missing[mb_idx, :]
            H_mb1 = self._sample_M(self.batch_size, self.dim, 1-self.hint_rate)
            H_mb = M_mb*H_mb1 + 0.5*(1-H_mb1)

            New_X_mb = M_mb * X_mb + (1-M_mb)*Z_mb #introduce missing data

            X_mb = torch.tensor(X_mb, dtype=torch.float32)
            New_X_mb = torch.tensor(New_X_mb, dtype=torch.float32)
            Z_mb = torch.tensor(Z_mb, dtype=torch.float32)
            M_mb = torch.tensor(M_mb, dtype=torch.float32)
            H_mb = torch.tensor(H_mb, dtype=torch.float32)

            #Train Discriminator
            G_sample = self.modelG(X_mb, New_X_mb, M_mb)
            D_prob = self.modelD(X_mb, M_mb, G_sample, H_mb)
            D_loss = bce_loss(D_prob, M_mb)

            D_loss.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            #Train Generator
            G_sample = self.modelG(X_mb, New_X_mb, M_mb)
            D_prob = self.modelD(X_mb, M_mb, G_sample, H_mb)
            D_prob.cpu().detach()
            G_loss1 = ((1-M_mb)*(torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb).sum()
            G_mse_loss = mse_loss(M_mb*X_mb, M_mb*G_sample)/M_mb.sum()
            G_loss = G_loss1 + self.alpha*G_mse_loss

            G_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            G_mse_test = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample)/(1-M_mb).sum()

            '''if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss))
                print('Train_loss: {:.4}'.format(G_mse_loss))
                print('Test_loss: {:.4}'.format(G_mse_test))
                print()'''
        self._Gen_params = self.modelG.state_dict()

        Z_mb = self._sample_Z(no, self.dim) 
        M_mb = Missing
        X_mb = norm_data_filled
   
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        X_mb = torch.tensor(X_mb, dtype=torch.float32)
        New_X_mb = torch.tensor(New_X_mb, dtype=torch.float32)
        M_mb = torch.tensor(M_mb, dtype=torch.float32)

        G_sample = self.modelG(X_mb, New_X_mb, M_mb)
        mse_final = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample)/(1-M_mb).sum()
        #print('Final Train RMSE: ' + str(np.sqrt(mse_final.item())))

        imputed_data = M_mb * X_mb + (1-M_mb) * G_sample
        imputed_data = imputed_data.cpu().detach().numpy()
        _, dim = imputed_data.shape
        renorm_data = imputed_data.copy()
        for i in range(dim):
            renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
            renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        for i in range(dim):
            temp = X[~np.isnan(X[:, i]), i]
            # Only for the categorical variable
            if len(np.unique(temp)) < 20:
                renorm_data[:, i] = np.round(renorm_data[:, i])
        return renorm_data
    
    def _sample_M(self, rows, cols, p):
        '''Sample binary random variables.
        Args:
            - p: probability of 1
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - binary_random_matrix: generated binary random matrix.
        '''
        unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
        binary_random_matrix = unif_random_matrix > p
        return 1.*binary_random_matrix

    def _sample_Z(self, rows, cols):
        '''Sample uniform random variables.
        Args:
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - uniform_random_matrix: generated uniform random matrix.
        '''
        return np.random.uniform(0., 1., size = [rows, cols])       

    def _sample_index(self, rows, batch_size):
        '''Sample index of the mini-batch.
        Args:
            - total: total number of samples (rows)
            - batch_size: batch size
        Returns:
            - batch_idx: batch index
        '''
        total_idx = np.random.permutation(rows)
        batch_idx = total_idx[:batch_size]
        return batch_idx
    
    class Generator(torch.nn.Module):
        def __init__(self, GainImputer):
            super(GainImputer.Generator, self).__init__()
            self.G1 = torch.nn.Linear(GainImputer.dim*2,GainImputer.int_dim)
            self.G2 = torch.nn.Linear(GainImputer.int_dim,GainImputer.int_dim)
            self.G3 = torch.nn.Linear(GainImputer.int_dim,GainImputer.dim)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.init_weight()

        def init_weight(self):
            layers = [self.G1, self.G2, self.G3]
            [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

        def forward(self, X: torch.float32, Z: torch.float32, M: torch.float32):
            input = M * X + (1-M)*Z
            input = torch.cat([input, M], dim=1)
            out = self.relu(self.G1(input))
            out = self.relu(self.G2(out))
            out = self.sigmoid(self.G3(out))
            return out
        
    class Discriminator(torch.nn.Module):
        def __init__(self, GainImputer):
            super(GainImputer.Discriminator, self).__init__()
            self.D1 = torch.nn.Linear(GainImputer.dim*2,GainImputer.int_dim)
            self.D2 = torch.nn.Linear(GainImputer.int_dim,GainImputer.int_dim)
            self.D3 = torch.nn.Linear(GainImputer.int_dim,GainImputer.dim)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.init_weight()
        
        def init_weight(self):
            layers = [self.D1, self.D2, self.D3]
            [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        def forward(self, X, M, G, H):
            input = M * X + (1-M)*G
            input = torch.cat([input, H], dim=1)
            out = self.relu(self.D1(input))
            out = self.relu(self.D2(out))
            out = self.D3(out)
            return out
