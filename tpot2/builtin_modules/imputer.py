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
from sklearn.utils._param_validation import _MissingValues
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
                 missing_values=np.nan, 
                 n_jobs=-1,
                 random_state=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.missing_values = missing_values
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
        torch.set_default_device(self.device)
        

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, )
        force_all_finite = False if self.missing_values in ["NaN", 
                            np.nan] else True
        if hasattr(X, 'dtypes'):
            X = X.to_numpy()
        #define mask matrix
        X_mask = 1 - np.isnan(X)
        #get dimensions
        no, dim = X.shape
        int_dim = int(dim)
        #normalize the original data, and save parameters for renormalization
        norm_data = X.copy()
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[i])
            norm_data[:, i] -= np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[i])
            norm_data[:, i] /= (np.nanmax(norm_data[:, i]) + 1e-06)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
        norm_data_filled = np.nan_to_num(norm_data, 0)

        Z_mb = self.uniform_sampler(0, 0.01, no, dim)
        M_mb = X_mask
        X_mb = norm_data_filled
        X_mb = M_mb * X_mb + (1 - M_mb)*Z_mb

        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        New_X_mb = torch.tensor(New_X_mb)
        #test loss
        G_sample = self.modelG.G_prob(New_X_mb, M_mb)
        MSE_final = torch.mean(((1-M_mb)*X_mb-(1-M_mb)*G_sample)**2)/ torch.mean(1-M_mb)
        print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))
        imputed_data = M_mb * X_mb + (1-M_mb) * G_sample
        imputed_data = imputed_data.detach().numpy
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
        
    def fit_transform(self, X, y=None):
        if hasattr(X, 'dtypes'):
            X = X.to_numpy()
        #define mask matrix
        X_mask = 1 - np.isnan(X)
        #get dimensions
        no, dim = X.shape
        int_dim = int(dim)
        #normalize the original data, and save parameters for renormalization
        norm_data = X.copy()
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[i])
            norm_data[:, i] -= np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[i])
            norm_data[:, i] /= (np.nanmax(norm_data[:, i]) + 1e-06)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
        norm_data_filled = np.nan_to_num(norm_data, 0)
        #Torch version of Gain
        # Initalize discriminator weights, gives hints and data as inputs
        D_W1 = torch.tensor(self._xavier_init([dim*2, int_dim])) 
        D_b1 = torch.zeros(size=[int_dim])
        D_W2 = torch.tensor(self._xavier_init([int_dim, int_dim]))
        D_b2 = torch.zeros(size=[int_dim])
        D_W3 = torch.tensor(self._xavier_init([int_dim, dim]))
        D_b3 = torch.zeros(size=[dim])
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        self.modelD = self.Discriminator(theta_D).to(self.device)
        # Initalize generator weights, gives hints and data as inputs
        G_W1 = torch.tensor(self._xavier_init([dim*2, int_dim])) 
        G_b1 = torch.zeros(size=[int_dim])
        G_W2 = torch.tensor(self._xavier_init([int_dim, int_dim]))
        G_b2 = torch.zeros(size=[int_dim])
        G_W3 = torch.tensor(self._xavier_init([int_dim, dim]))
        G_b3 = torch.zeros(size=[dim])
        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
        self.modelG = self.Generator(theta_G).to(self.device)
        #Data + Mask as inputs (Random noise is in missing components)
        def discriminator_loss(M, New_X, H):
            G_sample = self.modelG.G_prob(New_X, M)
            Hat_New_X = New_X * M + G_sample * (1-M)
            D_prob = self.modelD.D_prob(Hat_New_X, H)
            D_loss = -torch.mean(M*torch.log(D_prob + 1e-8) 
                                 + (1-M)*torch.log(1. - D_prob + 1e-8))
            return D_loss
        def generator_loss(X, M, New_X, H):
            G_sample = self.modelG.G_prob(New_X, M)
            Hat_New_X = New_X * M + G_sample * (1-M)
            D_prob = self.modelD.D_prob(Hat_New_X, H)
            G_loss = -torch.mean((1-M)*torch.log(D_prob + 1e-8))
            MSE_train_loss = torch.mean(
                (M*New_X - M*G_sample)**2
                ) / torch.mean(M)
            G_loss_final = G_loss + self.alpha*MSE_train_loss
            MSE_test_loss = torch.mean(
                ((1-M)*X-(1-M)*G_sample)**2
                )/ torch.mean(1-M)
            return G_loss_final, MSE_train_loss, MSE_test_loss
        optimizer_D = torch.optim.Adam(params=theta_D)
        optimizer_G = torch.optim.Adam(params=theta_G)
        #Training Iterations
        for it in range(self.iterations):
            batch_idx = self._sample_batch_index(no, self.batch_size)
            X_mb = norm_data_filled[batch_idx, :]
            M_mb = X_mask[batch_idx, :]
            #sample random vectors
            Z_mb = self._uniform_sampler(0, 0.01, self.batch_size, dim)
            #Sample hint vectors
            H_mb_temp = self._binary_sampler(self.hint_rate, 
                                             self.batch_size, dim)
            H_mb = M_mb * H_mb_temp
            #combine vectors with observed vectors
            X_mb = M_mb*X_mb + (1-M_mb)*Z_mb #Introduce Missin Data

            X_mb = torch.tensor(X_mb)
            M_mb = torch.tensor(M_mb)
            H_mb = torch.tensor(H_mb)
            New_X_mb = torch.tensor(New_X_mb)

            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
            G_loss_curr.backward()
            optimizer_G.step()

        Z_mb = self.uniform_sampler(0, 0.01, no, dim)
        M_mb = X_mask
        X_mb = norm_data_filled
        X_mb = M_mb * X_mb + (1 - M_mb)*Z_mb

        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        New_X_mb = torch.tensor(New_X_mb)
        #test loss
        G_sample = self.modelG.G_prob(New_X_mb, M_mb)
        MSE_final = torch.mean(((1-M_mb)*X_mb-(1-M_mb)*G_sample)**2)/ torch.mean(1-M_mb)
        print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))
        imputed_data = M_mb * X_mb + (1-M_mb) * G_sample
        imputed_data = imputed_data.detach().numpy
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








        return 

    def _binary_sampler(self, p, rows, cols):
        '''Sample binary random variables.
        Args:
            - p: probability of 1
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - binary_random_matrix: generated binary random matrix.
        '''
        unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
        binary_random_matrix = 1*(unif_random_matrix < p)
        return binary_random_matrix

    def _uniform_sampler(self, low, high, rows, cols):
        '''Sample uniform random variables.
        Args:
            - low: low limit
            - high: high limit
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - uniform_random_matrix: generated uniform random matrix.
        '''
        return np.random.uniform(low, high, size = [rows, cols])       

    def _sample_batch_index(self, total, batch_size):
        '''Sample index of the mini-batch.
        Args:
            - total: total number of samples
            - batch_size: batch size
        Returns:
            - batch_idx: batch index
        '''
        total_idx = np.random.permutation(total)
        batch_idx = total_idx[:batch_size]
        return batch_idx
    
    class Generator(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.theta_G = params
            self.G_W1 = self.theta_G[0] 
            self.G_b1 = self.theta_G[3]
            self.G_W2 = self.theta_G[1] 
            self.G_b2 = self.theta_G[4]
            self.G_W3 = self.theta_G[2]
            self.G_b3 = self.theta_G[5]

        def G_prob(self, X: torch.float32, M: torch.float32):
            inputs = torch.concat([X, M], dim=1)
            G_h1 = torch.nn.functional.relu(
                torch.matmul(inputs, self.G_W1) + self.G_b1
                )
            G_h2 = torch.nn.functional.relu(
                torch.matmul(G_h1, self.G_W2) + self.G_b2
                )
            g_prob = torch.nn.functional.sigmoid(
                torch.matmul(G_h2, self.G_W3) + self.G_b3
                )
            return g_prob

        def _xavier_init(self, size):
            '''Xavier initialization.
            Args:
                - size: vector size
            Returns:
                - initialized random vector.
            '''
            in_dim = size[0]
            xavier_stddev = 1./torch.sqrt(in_dim / 2.)
            return torch.normal(std = xavier_stddev, size=size)
        
    class Discriminator(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.theta_D = params
            self.D_W1 = self.theta_D[0] 
            self.D_b1 = self.theta_D[3]
            self.D_W2 = self.theta_D[1] 
            self.D_b2 = self.theta_D[4]
            self.D_W3 = self.theta_D[2]
            self.D_b3 = self.theta_D[5]
        
        def D_prob(self, X: torch.float32, H: torch.float32):
            inputs = torch.concat([X, H], dim=1)
            D_h1 = torch.nn.functional.relu(
                torch.matmul(inputs, self.D_W1) + self.D_b1
                )
            D_h2 = torch.nn.functional.relu(
                torch.matmul(D_h1, self.D_W2) + self.D_b2
                )
            d_prob = torch.nn.functional.sigmoid(
                torch.matmul(D_h2, self.D_W3) + self.D_b3
                )
            return d_prob

        def _xavier_init(self, size):
            '''Xavier initialization.
            Args:
                - size: vector size
            Returns:
                - initialized random vector.
            '''
            in_dim = size[0]
            xavier_stddev = 1./torch.sqrt(in_dim / 2.)
            return torch.normal(std = xavier_stddev, size=size)

        
    
                
        

        
                    
        
        


