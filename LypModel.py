import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import levy_stable, cauchy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['FLAGS_logtostderr'] = '0'     # disables logging to stderr
os.environ['FLAGS_minloglevel'] = '3'     # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0' # disables verbose logging

from keras.utils import to_categorical

import matplotlib.pyplot as plt

def levy_stable_sample(size, g, alpha=1, beta=0):
    """
    Generates samples from a Lévy alpha-stable distribution using scipy.
    alpha: stability parameter (0 < alpha <= 2)
    beta: skewness parameter (-1 <= beta <= 1)
    size: shape of the output tensor
    
    alpha   beta    equivalent
    1/2     -1      levy_l
    1/2     1       levy
    1       0       cauchy
    2       any     norm (with scale=sqrt(2)) [no heavy tail
    
    Stability parameter (alpha)
    The tails decay polynomially as |x|^{-alpha-1},
        * for alpha=1, the tails decay as |x|^{-2} (Cauchy distribution)
        * for alpha=0.5, the tails decay as |x|^{-1.5}
        * for alpha = 2, the distribution becomes Gaussian, without heavy tails
        
    Skewness Prameter (beta)
    Controls the asymmetry of the distribution
        * beta = 0 results in symmetric distribution
        * beta = -1: left skewed
        * beta = 1: right skewed
    This, however, doesn't directly affect the heaviness of the tails.
    
    # use centered distribution for now, diff beta for later investigation
    """
    hidden_size, _ = size
    samples = levy_stable.rvs(alpha, beta, size=size)
    scaled_samples = (g * samples) / (hidden_size ** (1/alpha))
    return torch.tensor(scaled_samples, dtype=torch.get_default_dtype()).to('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.ModuleList):
    def __init__(self, initialization='normal', input_size=100, hidden_size=1000, 
                 g=1, nonlinearity='tanh', alpha=1,
                 task=False, num_layers=1, output_size=82, p_drop=0.5, embedded=None,
                 data=None, batch_first=True):
        super(RNNModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.g = g
        self.alpha = alpha
        self.L = self.num_layers*self.hidden_size
        self.initialization = initialization
        self.rnn_layer = nn.RNN(input_size, hidden_size, num_layers, 
                                nonlinearity=nonlinearity,
                                batch_first=batch_first).to(self.device)
        
        self.task = task
        if self.task:
            self.dropout = nn.Dropout(p=p_drop)
            self.fc = nn.Linear(in_features = hidden_size, out_features = output_size, bias = False)
            
            self.embedded = embedded
            if self.embedded == None:
                self.encoder = nn.Identity()
            elif self.embedded == 'embedding':
                self.encoder = nn.Embedding(self.input_size, self.input_size)
            elif self.embedded == 'one-hot':
                self.encoder = lambda xt: torch.from_numpy(to_categorical(xt.cpu(), self.input_size)).to(self.device)
            else:
                raise Exception('Embedding type not defined')

        self._init_weights(data=data)
        
    def get_weight_spectrum(self):
        for name, param in self.rnn_layer.named_parameters():
            if 'weight_hh' in name:
                weight_matrix = param.detach().cpu().numpy()
                break
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(weight_matrix)
        return eigenvalues
        
    def _init_weights(self, data=None):
        size_normalization = 1.0 / np.sqrt(self.hidden_size)
        for name, param in self.rnn_layer.named_parameters():
            if 'weight_ih' in name:
                nn.init.normal_(param, mean=0.0, std=size_normalization)
            elif 'weight_hh' in name:
                if self.initialization == 'normal':
                    nn.init.normal_(param, mean=0.0, std=np.sqrt(2) * self.g * size_normalization)
                elif self.initialization == 'orthogonal':
                    nn.init.orthogonal_(param)
                    with torch.no_grad():
                        param *= self.g
                elif self.initialization == 'levy':
                    # breakpoint()
                    with torch.no_grad():
                        param.copy_(levy_stable_sample(g=self.g, alpha=self.alpha, beta=0, size=param.size()))
                        # param.copy_(cauchy_sample(g=self.g, size=param.size()))
                    # breakpoint()
                elif self.initialization in ['connectome', 'balanced_connectome', 'balanced_generated_connectome']:
                    print('Loading data for', self.initialization)
                    assert data is not None, 'data not loaded'
                    assert self.hidden_size == data.shape[0], 'data and model size mismatched'
                    with torch.no_grad():
                        param.copy_(self.g*torch.tensor(data, device=self.device))
                else:
                    assert 'initialization not found'
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                assert f'{name} is not reinitializaed!'

    def forward(self, xt, h):
        # breakpoint()
        if self.task:
            xt = self.encoder(xt)
        self.rnn_layer.flatten_parameters()
        rnn_out, rnn_hn = self.rnn_layer(xt, h)
        
        if self.task:
            d_out = self.dropout(rnn_out)
            output = self.fc(d_out)
            return output, rnn_hn
        else:
            return rnn_out, rnn_hn

    def init_hidden(self, batch_size):
        # h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        h = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.get_default_dtype())).to(self.device)
        return h
    
    def make_autonomous(self):
        """
        Shrink input dimensionality and zero-out input weights.
        Call *before* you send the model to the device.
        """
        assert self.input_size == 1, "build the RNN with input_size=1 for autonomous mode"
        with torch.no_grad():
            self.rnn_layer.weight_ih_l0.zero_()   # no input drive
            # (bias_ih_l0 is already zero in your init)
            # print('Zeroed input weights.')

    def make_noise_driven(self):
        """
        Per-unit i.i.d. Gaussian drive: input_size == hidden_size and weight_ih = I.
        """
        H = self.hidden_size
        assert self.input_size == H, "build the RNN with input_size==hidden_size for noise mode"
        with torch.no_grad():
            W = torch.eye(H, dtype=self.rnn_layer.weight_ih_l0.dtype,
                        device=self.rnn_layer.weight_ih_l0.device)
            self.rnn_layer.weight_ih_l0.copy_(W)   # I_H
            # Optional: stop gradients so the identity stays fixed
            self.rnn_layer.weight_ih_l0.requires_grad_(False)
            # print('Set input weights to identity.')

    
