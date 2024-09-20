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
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.compose
import torch
from torch.utils.data import DataLoader, TensorDataset

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
            np.random.seed(self.random_state)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X, y = None):
        
        self.modelG.load_state(self._Gen_params)

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

        G_sample = self.modelG.forward(X_mb, New_X_mb, M_mb)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        mse_final = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample)/(1-M_mb).sum()
        #print('Transform RMSE: ' + str(np.sqrt(mse_final.item())))

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
        
    def fit_transform(self, X, y=None):
        #print("working")
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
            G_sample = self.modelG.forward(X_mb, New_X_mb, M_mb)
            D_prob = self.modelD.forward(X_mb, M_mb, G_sample, H_mb)
            D_loss = bce_loss(D_prob, M_mb)

            D_loss.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            #Train Generator
            G_sample = self.modelG.forward(X_mb, New_X_mb, M_mb)
            D_prob = self.modelD.forward(X_mb, M_mb, G_sample, H_mb)
            D_prob.cpu().detach()
            G_loss1 = ((1-M_mb)*(torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb).sum()
            G_mse_loss = mse_loss(M_mb*X_mb, M_mb*G_sample)/M_mb.sum()
            G_loss = G_loss1 + self.alpha*G_mse_loss

            G_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            G_mse_test = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample)/(1-M_mb).sum()
            '''
            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss))
                print('Train_loss: {:.4}'.format(G_mse_loss))
                print()
            '''
        self._Gen_params = self.modelG.parameters()

        Z_mb = self._sample_Z(no, self.dim) 
        M_mb = Missing
        X_mb = norm_data_filled
   
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        X_mb = torch.tensor(X_mb, dtype=torch.float32)
        New_X_mb = torch.tensor(New_X_mb, dtype=torch.float32)
        M_mb = torch.tensor(M_mb, dtype=torch.float32)

        G_sample = self.modelG.forward(X_mb, New_X_mb, M_mb)
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
    
    class Generator():
        def __init__(self, GainImputer):
            super(GainImputer.Generator, self).__init__()
            self.G_W1 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.int_dim, GainImputer.dim*2), requires_grad=True, device=GainImputer.device))    # Data + Hint as inputs
            self.G_b1 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)

            self.G_W2 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.int_dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.G_b2 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)

            self.G_W3 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.G_b3 = torch.zeros((GainImputer.dim), requires_grad=True, device=GainImputer.device)   

        def forward(self, X: torch.float32, Z: torch.float32, M: torch.float32):
            input = M * X + (1-M)*Z
            input = torch.cat([input, M], dim=1)
            l1 = torch.nn.functional.linear(input=input, weight=self.G_W1, bias=self.G_b1)
            out1 = torch.nn.functional.relu(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.G_W2, bias=self.G_b2)
            out2 = torch.nn.functional.relu(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.G_W3, bias=self.G_b3)
            out = torch.nn.functional.sigmoid(l3)
            return out
        
        def parameters(self):
            params = [self.G_W1, self.G_b1, self.G_W2, self.G_b2, self.G_W3, self.G_b3]
            return params
        
        def load_state(self, params):
            self.G_W1 = params[0]
            self.G_b1 = params[1]
            self.G_W2 = params[2]
            self.G_b2 = params[3]
            self.G_W3 = params[4]
            self.G_b3 = params[5]
        
    class Discriminator():
        def __init__(self, GainImputer):
            super(GainImputer.Discriminator, self).__init__()
            self.D_W1 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.int_dim, GainImputer.dim*2), requires_grad=True, device=GainImputer.device))     # Data + Hint as inputs
            self.D_b1 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)
            self.D_W2 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.int_dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.D_b2 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)
            self.D_W3 = torch.nn.init.kaiming_normal_(torch.empty((GainImputer.dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.D_b3 = torch.zeros((GainImputer.dim), requires_grad=True, device=GainImputer.device)       # Output is multi-variate

        
        def forward(self, X, M, G, H):
            input = M * X + (1-M)*G
            input = torch.cat([input, H], dim=1)
            l1 = torch.nn.functional.linear(input=input, weight=self.D_W1, bias=self.D_b1)
            out1 = torch.nn.functional.relu(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.D_W2, bias=self.D_b2)
            out2 = torch.nn.functional.relu(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.D_W3, bias=self.D_b3)
            return l3
        
        def parameters(self):
            params = [self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3]
            return params

class VAEImputer(BaseEstimator, TransformerMixin):

    def __init__(self, iterations=1000, batch_size=128, split_size=5, code_size=5, encoder_hidden_sizes=[128, 64], decoder_hidden_sizes=[128, 64],
                    temperature=None, p_miss = 0.2, learning_rate = 0.001, tolerance=0.001, random_state=None):
        
        self.batch_size = batch_size
        self.iterations = iterations
        self.split_size = split_size
        self.code_size = code_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.test_loss_function = torch.nn.MSELoss()
        self.p_miss = p_miss
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.random_state=random_state
        torch.set_default_dtype(torch.float32)
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
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    class Encoder():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.Encoder, self).__init__()
            self.E_W1 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.encoder_hidden_sizes[0], input_size), requires_grad=True, device=VAEImputer.device))    # Data + Hint as inputs
            self.E_b1 = torch.zeros((VAEImputer.encoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device)

            self.E_W2 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.encoder_hidden_sizes[1], VAEImputer.encoder_hidden_sizes[0],),requires_grad=True, device=VAEImputer.device))
            self.E_b2 = torch.zeros((VAEImputer.encoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device)

            self.E_W3 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.code_size, VAEImputer.encoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device))
            self.E_b3 = torch.zeros((VAEImputer.code_size), requires_grad=True, device=VAEImputer.device)   
        
            self.E_W4 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.split_size, VAEImputer.code_size),requires_grad=True, device=VAEImputer.device))
            self.E_b4 = torch.zeros((VAEImputer.split_size), requires_grad=True, device=VAEImputer.device)   

            self.E_W5 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.split_size, VAEImputer.code_size),requires_grad=True, device=VAEImputer.device))
            self.E_b5 = torch.zeros((VAEImputer.split_size), requires_grad=True, device=VAEImputer.device)  

        def forward(self, x):
            l1  = torch.nn.functional.linear(input=x, weight=self.E_W1, bias=self.E_b1)
            out1 = torch.nn.functional.tanh(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.E_W2, bias=self.E_b2)
            out2 = torch.nn.functional.tanh(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.E_W3, bias=self.E_b3)
            out3 = torch.nn.functional.tanh(l3)
            mean = torch.nn.functional.linear(input=out3, weight=self.E_W4, bias=self.E_b4)
            log_var = torch.nn.functional.linear(input=out3, weight=self.E_W5, bias=self.E_b5)
            return mean, log_var

        def parameters(self):
            params = [self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3, self.E_W4, self.E_b4, self.E_W5, self.E_b5]
            return params
        
        def load_state(self, params):
            self.E_W1 = params[0]
            self.E_b1 = params[1]
            self.E_W2 = params[2]
            self.E_b2 = params[3]
            self.E_W3 = params[4]
            self.E_b3 = params[5]
            self.E_W4 = params[6]
            self.E_b4 = params[7]
            self.E_W5 = params[8]
            self.E_b5 = params[9]

    class Decoder():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.Decoder, self).__init__()
            self.D_W1 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.decoder_hidden_sizes[0], VAEImputer.split_size), requires_grad=True, device=VAEImputer.device))    # Data + Hint as inputs
            self.D_b1 = torch.zeros((VAEImputer.decoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device)

            self.D_W2 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.decoder_hidden_sizes[1], VAEImputer.decoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device))
            self.D_b2 = torch.zeros((VAEImputer.decoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device)

            self.D_W3 = torch.nn.init.xavier_normal_(torch.empty((input_size, VAEImputer.decoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device))
            self.D_b3 = torch.zeros((input_size), requires_grad=True, device=VAEImputer.device)   

        def forward(self, x):
            l1  = torch.nn.functional.linear(input=x, weight=self.D_W1, bias=self.D_b1)
            out1 = torch.nn.functional.tanh(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.D_W2, bias=self.D_b2)
            out2 = torch.nn.functional.tanh(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.D_W3, bias=self.D_b3)
            x_hat = torch.nn.functional.sigmoid(l3)
            return x_hat

        def parameters(self):
            params = [self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3]
            return params
        
        def load_state(self, params):
            self.D_W1 = params[0]
            self.D_b1 = params[1]
            self.D_W2 = params[2]
            self.D_b2 = params[3]
            self.D_W3 = params[4]
            self.D_b3 = params[5]
            
    class VAE():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.VAE, self).__init__()
            self.encoder = VAEImputer.Encoder(VAEImputer, input_size)
            self.decoder = VAEImputer.Decoder(VAEImputer, input_size)

        def forward(self, x):
            mu, log_var = self.encoder.forward(x)
            code = self.reparameterize(mu, log_var)
            reconstucted = self.decoder.forward(code)
            return code, reconstucted, mu, log_var

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
            
        def parameters(self):
            params = self.encoder.parameters() + self.decoder.parameters()
            return params
        
        def load_state(self, params):
            self.encoder.load_state(params[0:10]) 
            self.decoder.load_state(params[10:]) 

    def fit(self, X, y=None):
        #print('working')
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X
        
        self.encoder_hidden_sizes = [int(math.floor(X.shape[1]/2)), int(math.floor(X.shape[1]*3/10))]
        self.decoder_hidden_sizes = [int(math.floor(X.shape[1]*3/10)), int(math.floor(X.shape[1]/2))]
        self.split_size =int(math.floor(X.shape[1]/5))
        self.code_size=int(math.floor(X.shape[1]/5))
        
        #print(self.encoder_hidden_sizes)

        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)
        features = features.to(dtype=torch.float32)
        features = features.to(device=self.device)
        

        num_samples = len(features)
        variable_masks = []
        for variable_size in self.variable_sizes:
            variable_mask = (torch.zeros(num_samples, 1).uniform_(0.0, 1.0) > self.p_miss).float()
            if variable_size > 1:
                variable_mask = variable_mask.repeat(1, variable_size)
            variable_masks.append(variable_mask)
        mask = torch.cat(variable_masks, dim=1)

        temperature = self.temperature
        self.model = self.VAE(VAEImputer=self, input_size=features.shape[1])
        
        inverted_mask = 1 - mask
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask

        if self.learning_rate is not None:
            missing = torch.autograd.Variable(missing, requires_grad=True)
            self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.learning_rate)

        #pbar = tqdm(range(self.iterations))
        for iterations in range(self.iterations):
            train_ds = torch.utils.data.TensorDataset(features.float(), mask.float(), noisy_features.float())
            losses = [np.inf]
            for f, m, n in torch.utils.data.DataLoader(train_ds, 
                                                       batch_size=self.batch_size,
                                                         shuffle=True, 
                                                         generator=torch.Generator(device=self.device)):
                loss = self.train_batch(f, m, n)
                temp_loss = losses[-1]
                
                if temp_loss - loss < self.tolerance:
                    break
                
                losses.append(loss)
            #pbar.set_postfix({'loss': min(losses)})
            '''
            if iterations % 100 == 0 :
                print(f'Epoch {iterations} loss: {loss:.4f}')
            '''

        self._VAE_params = self.model.parameters()
        #print(len(self._VAE_params))
        return self
    
    def train_batch(self, features, mask, noisy_features):
        self.optim.zero_grad()
        #print(features.shape)
        #print(noisy_features.shape)
        #noise = torch.autograd.Variable(torch.FloatTensor(len(noisy_features), self.p_miss).normal_())
        _, reconstructed, mu, log_var = self.model.forward(noisy_features)
        #print(reconstructed.shape)
        #print(reconstructed)
        # reconstruction of the non-missing values
        reconstruction_loss = self.masked_reconstruction_loss_function(reconstructed,
                                                                  features,
                                                                  mask,
                                                                  self.variable_sizes)
        missing_loss = self.masked_reconstruction_loss_function(reconstructed, features, 1-mask, self.variable_sizes)
        #print(reconstruction_loss)
        loss = torch.sqrt(self.test_loss_function((mask * features + (1.0 - mask) * reconstructed), features))
        
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #print(kld_loss)
        observed_loss = reconstruction_loss + kld_loss
        #loss = loss.type(torch.float32)
        #print(loss)
        observed_loss.backward()

        self.optim.step()

        return observed_loss.cpu().detach().numpy()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        self.model.load_state(self._VAE_params)
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X
        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)
        mask = torch.from_numpy(1-np.isnan(X.to_numpy()))
        inverted_mask = ~mask
        num_samples = len(features)
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask
        
        f = features.to(dtype=torch.float32)
        f = f.to(device=self.device)
        
        m = mask.to(dtype=torch.float32)
        m = m.to(device=self.device)
        #print(m)
        n = noisy_features.to(dtype=torch.float32)
        n = n.to(device=self.device)
        #print(n)
        with torch.no_grad():
            _, reconstructed, _, _ = self.model.forward(n)
            #print(reconstructed)
            imputed = m*n + (1.0 - m)*reconstructed
        return imputed.cpu().numpy()

    def reconstruction_loss_function(self, reconstructed, original, variable_sizes, reduction="mean"):
        # by default use loss for binary variables
        if variable_sizes is None:
            return torch.nn.functional.binary_cross_entropy(reconstructed, original, reduction=reduction)
        # use the variable sizes when available
        else:
            loss = 0
            start = 0
            numerical_size = 0
            for variable_size in variable_sizes:
                # if it is a categorical variable
                if variable_size > 1:
                    # add loss from the accumulated continuous variables
                    if numerical_size > 0:
                        end = start + numerical_size
                        batch_reconstructed_variable = reconstructed[:, start:end]
                        batch_target = original[:, start:end]
                        loss += torch.nn.functional.mse_loss(batch_reconstructed_variable, batch_target, reduction=reduction)
                        start = end
                        numerical_size = 0
                    # add loss from categorical variable
                    end = start + variable_size
                    batch_reconstructed_variable = reconstructed[:, start:end]
                    batch_target = torch.argmax(original[:, start:end], dim=1)
                    loss += torch.nn.functional.cross_entropy(batch_reconstructed_variable, batch_target, reduction=reduction)
                    start = end
                # if not, accumulate numerical variables
                else:
                    numerical_size += 1

            # add loss from the remaining accumulated numerical variables
            if numerical_size > 0:
                end = start + numerical_size
                batch_reconstructed_variable = reconstructed[:, start:end]
                batch_target = original[:, start:end]
                loss += torch.nn.functional.mse_loss(batch_reconstructed_variable, batch_target, reduction=reduction)

            return loss

    def masked_reconstruction_loss_function(self, reconstructed, original, mask, variable_sizes):
        return self.reconstruction_loss_function(mask * reconstructed,
                                            mask * original,
                                            variable_sizes,
                                            reduction="sum") / torch.sum(mask)

