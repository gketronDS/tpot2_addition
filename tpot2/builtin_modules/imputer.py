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
        print('Transform RMSE: ' + str(np.sqrt(mse_final.item())))

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

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss))
                print('Train_loss: {:.4}'.format(G_mse_loss))
                print()
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
        print('Final Train RMSE: ' + str(np.sqrt(mse_final.item())))

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
            self.G_W1 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.int_dim, GainImputer.dim*2), requires_grad=True, device=GainImputer.device))    # Data + Hint as inputs
            self.G_b1 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)

            self.G_W2 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.int_dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.G_b2 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)

            self.G_W3 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
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
            self.D_W1 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.int_dim, GainImputer.dim*2), requires_grad=True, device=GainImputer.device))     # Data + Hint as inputs
            self.D_b1 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)
            self.D_W2 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.int_dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
            self.D_b2 = torch.zeros((GainImputer.int_dim),requires_grad=True, device=GainImputer.device)
            self.D_W3 = torch.nn.init.xavier_normal_(torch.empty((GainImputer.dim, GainImputer.int_dim),requires_grad=True, device=GainImputer.device))
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
                    temperature=None, p_miss = 0.2, learning_rate = 0.001, tolerance=0.001):
        
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
        torch.set_default_dtype(torch.float32)

    def fit(self, X, y=None):
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X
        
        self.encoder_hidden_sizes = [int(math.floor(X.shape[1]/2)), int(math.floor(X.shape[1]*3/10))]
        self.decoder_hidden_sizes = [int(math.floor(X.shape[1]/2)), int(math.floor(X.shape[1]*3/10))]
        self.split_size =int(math.floor(X.shape[1]/5))
        self.code_size=int(math.floor(X.shape[1]/5))
        
        #print(self.encoder_hidden_sizes)

        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)

        num_samples = len(features)
        variable_masks = []
        for variable_size in self.variable_sizes:
            variable_mask = (torch.zeros(num_samples, 1).uniform_(0.0, 1.0) > self.p_miss).float()
            if variable_size > 1:
                variable_mask = variable_mask.repeat(1, variable_size)
            variable_masks.append(variable_mask)
        mask = torch.cat(variable_masks, dim=1)

        temperature = self.temperature
        self.model = self.VAE(self,
                        features.shape[1],
                        self.split_size,
                        self.code_size,
                        encoder_hidden_sizes=self.encoder_hidden_sizes,
                        decoder_hidden_sizes=self.decoder_hidden_sizes,
                        variable_sizes=(None if temperature is None else self.variable_sizes),  # do not use multi-output without temperature
                        temperature=temperature
                        )
        
        self.model.train(mode=True)
        inverted_mask = 1 - mask
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask

        if self.learning_rate is not None:
            missing = torch.autograd.Variable(missing, requires_grad=True)
            self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.learning_rate)

        self.model.train(mode=True)
        #pbar = tqdm(range(self.iterations))
        for iterations in range(self.iterations):
            train_ds = TensorDataset(features.float(), mask.float(), noisy_features.float())
            losses = []
            for f, m, n in DataLoader(train_ds, batch_size=self.batch_size, shuffle=True):
                loss = self.train_batch(f, m, n)
                losses.append(loss)
                if loss < self.tolerance:
                    break
            #pbar.set_postfix({'loss': min(losses)})
            '''
            if iterations % 100 == 0 :
                print(f'Epoch {iterations} loss: {loss:.4f}')
            '''

        self._VAE_params = self.model.state_dict()
        return self
    
    def train_batch(self, features, mask, noisy_features):
        self.optim.zero_grad()
        #print(features.shape)
        #print(noisy_features.shape)
        #noise = torch.autograd.Variable(torch.FloatTensor(len(noisy_features), self.p_miss).normal_())
        _, reconstructed, mu, log_var = self.model(noisy_features, training=True)
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

        return observed_loss.detach().numpy()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        self.model.load_state_dict(self._VAE_params)
        self.model.train(mode=False)
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X

        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)
        mask = torch.from_numpy(1-np.isnan(X.to_numpy()))
        inverted_mask = ~mask
        num_samples = len(features)
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask
        f = features.float()
        m = mask.float()
        #print(m)
        n = noisy_features.float()
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

    class Encoder(torch.nn.Module):

        def __init__(self, VAEImputer, input_size, code_size, hidden_sizes=[], variable_sizes=None):
            super(VAEImputer.Encoder, self).__init__()

            layers = []

            if variable_sizes is None:
                previous_layer_size = input_size
                #print(type(previous_layer_size))
            else:
                multi_input_layer = VAEImputer.MultiInput(VAEImputer, variable_sizes)
                layers.append(multi_input_layer)
                previous_layer_size = multi_input_layer.size
                #print(type(previous_layer_size))

            layer_sizes = list(hidden_sizes) + [code_size]
            hidden_activation = torch.nn.Tanh()

            for layer_size in layer_sizes:
                #print(layer_size)
                layers.append(torch.nn.Linear(previous_layer_size, layer_size))
                layers.append(hidden_activation)
                previous_layer_size = layer_size

            self.hidden_layers = torch.nn.Sequential(*layers)

        def forward(self, inputs):
            #print(inputs)
            return self.hidden_layers(inputs)
    
    class Decoder(torch.nn.Module):

        def __init__(self, VAEImputer, code_size, output_size, hidden_sizes=[], variable_sizes=None, temperature=None):
            super(VAEImputer.Decoder, self).__init__()

            hidden_activation = torch.nn.Tanh()

            previous_layer_size = code_size
            hidden_layers = []

            for layer_size in hidden_sizes:
                hidden_layers.append(torch.nn.Linear(previous_layer_size, layer_size))
                hidden_layers.append(hidden_activation)
                previous_layer_size = layer_size

            if len(hidden_layers) > 0:
                self.hidden_layers = torch.nn.Sequential(*hidden_layers)
            else:
                self.hidden_layers = None

            if variable_sizes is None:
                self.output_layer = VAEImputer.SingleOutput(VAEImputer, previous_layer_size, output_size, activation=torch.nn.Sigmoid())
            else:
                self.output_layer = VAEImputer.MultiOutput(VAEImputer, previous_layer_size, variable_sizes, temperature=temperature)

        def forward(self, code, training=False):
            if self.hidden_layers is None:
                hidden = code
            else:
                hidden = self.hidden_layers(code)

            return self.output_layer(hidden, training=training)

    class VAE(torch.nn.Module):

        def __init__(self, VAEImputer, input_size, split_size, code_size, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                    variable_sizes=None, temperature=None):

            super(VAEImputer.VAE, self).__init__()

            self.encoder = VAEImputer.Encoder(VAEImputer, input_size, split_size, hidden_sizes=encoder_hidden_sizes, variable_sizes=variable_sizes)
            self.decoder = VAEImputer.Decoder(VAEImputer, code_size, input_size, hidden_sizes=decoder_hidden_sizes, variable_sizes=variable_sizes,
                                temperature=temperature)

            self.mu_layer = torch.nn.Linear(split_size, code_size)
            self.log_var_layer = torch.nn.Linear(split_size, code_size)

        def forward(self, inputs, training=False):
            mu, log_var = self.encode(inputs)
            #print(mu)
            #print(log_var)
            code = self.reparameterize(mu, log_var)
            #print(code.shape)
            reconstructed = self.decode(code, training=training)
            return code, reconstructed, mu, log_var

        def encode(self, inputs):
            outputs = self.encoder(inputs)
            #print(outputs.shape)
            #print(outputs)
            return self.mu_layer(outputs), self.log_var_layer(outputs)

        def decode(self, code, training=False):
            return self.decoder(code, training=training)
        
        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        
    '''
    class OutputLayer(torch.nn.Module):
        """
        This is just a simple abstract class for single and multi output layers.
        Both need to have the same interface.
        """

        def forward(self, hidden, training=None):
            raise NotImplementedError
    '''

    class SingleOutput(torch.nn.Module):

        def __init__(self, VAEImputer, previous_layer_size, output_size, activation=None):
            super(VAEImputer.SingleOutput, self).__init__()
            if activation is None:
                self.model = torch.nn.Linear(previous_layer_size, output_size)
            else:
                self.model = torch.nn.Sequential(torch.nn.Linear(previous_layer_size, output_size), activation)

        def forward(self, hidden, training=False):
            return self.model(hidden)
    
    class MultiOutput(torch.nn.Module):
        def __init__(self, VAEImputer, input_size, variable_sizes, temperature=None):
            super(VAEImputer.MultiOutput, self).__init__()

            self.output_layers = torch.nn.ModuleList()
            self.output_activations = torch.nn.ModuleList()

            numerical_size = 0
            for i, variable_size in enumerate(variable_sizes):
                # if it is a categorical variable
                if variable_size > 1:
                    # first create the accumulated numerical layer
                    if numerical_size > 0:
                        self.output_layers.append(torch.nn.Linear(input_size, numerical_size))
                        self.output_activations.append(VAEImputer.NumericalActivation())
                        numerical_size = 0
                    # create the categorical layer
                    self.output_layers.append(torch.nn.Linear(input_size, variable_size))
                    self.output_activations.append(VAEImputer.CategoricalActivation(temperature))
                # if not, accumulate numerical variables
                else:
                    numerical_size += 1

            # create the remaining accumulated numerical layer
            if numerical_size > 0:
                self.output_layers.append(torch.nn.Linear(input_size, numerical_size))
                self.output_activations.append(VAEImputer.NumericalActivation())

        def forward(self, inputs, training=True, concat=True):
            outputs = []
            for output_layer, output_activation in zip(self.output_layers, self.output_activations):
                logits = output_layer(inputs)
                output = output_activation(logits, training=training)
                outputs.append(output)

            if concat:
                return torch.cat(outputs, dim=1)
            else:
                return outputs


    class CategoricalActivation(torch.nn.Module):

        def __init__(self, VAEImputer, temperature):
            super(VAEImputer.CategoricalActivation, self).__init__()

            self.temperature = temperature

        def forward(self, logits, training=True):
            # gumbel-softmax (training and evaluation)
            if self.temperature is not None:
                return torch.nn.functional.gumbel_softmax(logits, hard=not training, tau=self.temperature)
            # softmax training
            elif training:
                return torch.nn.functional.softmax(logits, dim=1)
            # softmax evaluation
            else:
                return torch.distributions.OneHotCategorical(logits=logits).sample()


    class NumericalActivation(torch.nn.Module):

        def __init__(self, VAEImputer):
            super(VAEImputer.NumericalActivation, self).__init__()

        def forward(self, logits, training=True):
            return torch.sigmoid(logits)
        
    class MultiInput(torch.nn.Module):

        def __init__(self, VAEImputer, variable_sizes, min_embedding_size=2, max_embedding_size=50):
            super(VAEImputer.MultiInput, self).__init__()

            self.has_categorical = False
            self.size = 0

            embeddings = torch.nn.ParameterList()
            for i, variable_size in enumerate(variable_sizes):
                # if it is a numerical variable
                if variable_size == 1:
                    embeddings.append(None)
                    self.size += 1
                # if it is a categorical variable
                else:
                    # this is an arbitrary rule of thumb taken from several blog posts
                    embedding_size = max(min_embedding_size, min(max_embedding_size, int(variable_size / 2)))

                    # the embedding is implemented manually to be able to use one hot encoding
                    # PyTorch embedding only accepts as input label encoding
                    embedding = torch.nn.Parameter(data=torch.Tensor(variable_size, embedding_size).normal_(), requires_grad=True)

                    embeddings.append(embedding)
                    self.size += embedding_size
                    self.has_categorical = True

            if self.has_categorical:
                self.variable_sizes = variable_sizes
                self.embeddings = embeddings

        def forward(self, inputs):
            if self.has_categorical:
                outputs = []
                start = 0
                for variable_size, embedding in zip(self.variable_sizes, self.embeddings):
                    # extract the variable
                    end = start + variable_size
                    variable = inputs[:, start:end]

                    # numerical variable
                    if variable_size == 1:
                        # leave the input as it is
                        outputs.append(variable)
                    # categorical variable
                    else:
                        output = torch.matmul(variable, embedding).squeeze(1)
                        outputs.append(output)

                    # move the variable limits
                    start = end

                # concatenate all the variable outputs
                return torch.cat(outputs, dim=1)
            else:
                return inputs
        

