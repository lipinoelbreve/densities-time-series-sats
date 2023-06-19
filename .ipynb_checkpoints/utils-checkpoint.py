#%% IMPORTS

# General
import numpy as np
import pandas as pd
import datetime as dt
from time import sleep
import itertools
from pmdarima.arima import auto_arima
import json
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Scipy
from scipy.stats import norm
from scipy import integrate
from scipy.special import rel_entr

# Statsmodels
from statsmodels.regression import linear_model
import statsmodels.api as sm

# Parametric
from pomegranate import GeneralMixtureModel, NormalDistribution, LogNormalDistribution

# fPCA
import skfda
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline

# Sklearn
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

# Pytorch
import torch
import torch.nn as nn
from torch import optim as optim
from torch.utils.data import TensorDataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% PARAMETERS & CLASSES - GENERAL
ORANGE = plt.colormaps['Oranges']
BLUE = plt.colormaps['Blues']
VIRIDIS = plt.colormaps['viridis']

EPS = np.finfo(float).eps

def get_kl(f1, f2):
  KL = sum(
        rel_entr(
            f1 + EPS,
            f2 + EPS,
            )
        )
  return KL

def get_kdes(data, feature, group_var, time_periods):
  bandwidths = 1.06 * data.groupby(group_var, as_index=True)[feature].apply(lambda x: x.std() * (x.count() ** (-0.2)))
  kdes = dict()
  for t in time_periods:
    kde = KernelDensity(
      kernel='gaussian',
      bandwidth=bandwidths[t]).fit( data.loc[data[group_var]==t, feature].values.reshape(-1,1))

    kdes[t] = kde
  return kdes

class FitData():
  """
  Fits a regression over the train sample; then, predicts over the test sample
  
  method: 'lin-lin', 'log-lin', 'poly', 'logit'
  degree: integer greater than 1, only used when method == 'poly'
  X_train, y_train, X_test: array or DataFrame
  """
  def __init__(self, method='lin-lin', degree=2):
    self.method = method
    self.degree = degree

  def fit(self, X_train, y_train):
    self.original_X_train = X_train
    self.original_y_train = y_train

    if self.method == 'log-lin':
      self.y_train = y_train.copy()
      self.min_value = 0
      if any(self.y_train < 0):
        self.min_value = abs(self.y_train.min()) + 1
        self.y_train += self.min_value
        
      self.X_train = sm.add_constant( X_train )
      self.y_train = np.log( self.y_train )
    
    elif self.method == 'poly':
      res = X_train
      for i in np.arange(2, self.degree+1):
        res = np.column_stack((res, X_train**i))
      self.X_train = sm.add_constant( res )
      self.y_train = y_train

    elif self.method == 'logit':
      self.X_train = sm.add_constant( X_train )
      y_train = np.array( y_train )      
      self.y_train = np.log( y_train / ( 1 - y_train ) )
    
    else:
      self.X_train = sm.add_constant( X_train )
      self.y_train = y_train

    self.ols = sm.OLS(self.y_train, self.X_train)
    self.res = self.ols.fit()
    try:
      self.rlm = sm.RLM(self.y_train, self.X_train, M=sm.robust.norms.HuberT())
      self.rlm = self.rlm.fit()
    except:
      self.rlm = sm.RLM(self.y_train, self.X_train, M=sm.robust.norms.LeastSquares())
      self.rlm = self.rlm.fit()
    
  def predict(self, n_periods):
    T = self.original_X_train[-1] + 1
    X_test = range(T, T+n_periods)
    self.original_X_test = X_test
    
    if self.method == 'log-lin':
      self.X_test = sm.add_constant( X_test, has_constant='add' )
      self.prediction = self.res.get_prediction(self.X_test)  
      self.y_hat = np.exp( self.prediction.predicted_mean ) - self.min_value
      self.IC = self.prediction.conf_int(alpha=0.05)
      self.IC_lower = np.exp( self.IC[:,0] ) - self.min_value
      self.IC_upper = np.exp( self.IC[:,1] ) - self.min_value
      self.y_hat_robust = np.exp( self.rlm.predict(self.X_test) ) - self.min_value


    elif self.method == 'poly':
      res = X_test
      for i in np.arange(2, self.degree+1):
        res = np.column_stack((res, X_test**i))
      self.X_test = sm.add_constant( res, has_constant='add' )
      self.prediction = self.res.get_prediction(self.X_test)
      self.y_hat = self.prediction.predicted_mean
      self.IC = self.prediction.conf_int(alpha=0.05)
      self.IC_lower = self.IC[:,0]
      self.IC_upper = self.IC[:,1]
      self.y_hat_robust = self.rlm.predict(self.X_test)

    elif self.method == 'logit':
      self.X_test = sm.add_constant( X_test, has_constant='add' )
      self.prediction = self.res.get_prediction(self.X_test)
      self.y_hat = np.exp(self.prediction.predicted_mean)  / (1 + np.exp(self.prediction.predicted_mean))
      self.IC = self.prediction.conf_int(alpha=0.05)
      self.IC_lower = np.exp(self.IC[:,0]) / (1 + np.exp(self.IC[:,0]))
      self.IC_upper = np.exp(self.IC[:,1]) / (1 + np.exp(self.IC[:,1]))
      self.y_hat_robust = np.exp(self.rlm.predict(self.X_test)) / (1 + np.exp(self.rlm.predict(self.X_test)))

    else:
      self.X_test = sm.add_constant( X_test, has_constant='add' )
      self.prediction = self.res.get_prediction(self.X_test)
      self.y_hat = self.prediction.predicted_mean
      self.IC = self.prediction.conf_int(alpha=0.05)
      self.IC_lower = self.IC[:,0]
      self.IC_upper = self.IC[:,1]
      self.y_hat_robust = self.rlm.predict(self.X_test)

    return self.y_hat_robust

  def plot(self):
    plt.plot(self.original_X_train, self.original_y_train, label='actual')
    plt.plot(self.original_X_test, self.y_hat, label='forecast')
    plt.fill_between(self.original_X_test, self.IC_lower, self.IC_upper,
                    color='b', alpha=.15)
    plt.plot(self.original_X_test, self.y_hat_robust, label='robust')


#%% PARAMEMTERS & CLASSES - PARAMÉTRICO
class Mixture():
  def __init__(self, x, t, n_components, distributions=NormalDistribution):
    self.x = x
    self.n_components = n_components
    self.distributions = distributions
  
    self.model = np.nan
    for _ in range(1000):
      model = GeneralMixtureModel.from_samples(
        self.distributions,
        n_components=self.n_components,
        X=self.x.reshape(-1,1)
        )
      if pd.notna(model.probability(x).sum()):
        self.model = model
        break
    
    if pd.isna(self.model):
      print(f'Unable to fit distribution for t={t}')
    
    order = np.argsort([distribution.parameters[0] for distribution in self.model.distributions])
    self.weights = np.array(json.loads(self.model.to_json())['weights'])[order]
    self.fitted_distributions = self.model.distributions[ order ]

class Mixtures():
  def __init__(self, data, vars, n_components, distributions=NormalDistribution):
    self.data = data
    self.time_periods = pd.unique(self.data.time)
    self.vars = vars
    self.time_periods = pd.unique(self.data.time)
    self.T = self.data.time.max() + 1
    self.n_components = n_components
    self.distributions = distributions

  def fit(self):
    self.mixtures = [ Mixture(x=self.data[self.data.time==t][self.vars].values, t=t, n_components=self.n_components, distributions=self.distributions) \
                     for t in self.time_periods ]
    
    self.means = pd.DataFrame(
      [[d.parameters[0] for d in m.fitted_distributions] for m in self.mixtures],
      columns=['m_'+str(i) for i in range(self.n_components)],
      index=self.time_periods
    )
    
    self.deviations = pd.DataFrame(
      [[d.parameters[1] for d in m.fitted_distributions] for m in self.mixtures],
      columns=['d_'+str(i) for i in range(self.n_components)],
      index=self.time_periods
    )
    
    self.weights = pd.DataFrame(
      [m.weights for m in self.mixtures],
      columns=['w_'+str(i) for i in range(self.n_components)],
      index=self.time_periods
    )
    
  def forecast(self, steps, predict=['mean']):
    self.steps = steps
    
    # Means
    pred_mean = []
    for col in self.means:
      arima = auto_arima(self.means[col], start_p=0, d=1, start_q=0,
                          max_p=5, max_d=5, max_q=5, start_P=0,
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=1, seasonal=False,
                          error_action='warn', trace=False,
                          stepwise=True, random_state=20, n_fits=1000 )
      pred_mean.append( arima.predict(n_periods=steps) )

    self.pred_mean = pd.DataFrame(
        np.column_stack(pred_mean),
        columns=self.means.columns,
        index=range(self.time_periods[-1]+1 ,self.time_periods[-1]+steps+1)
    )

    # Deviations
    self.pred_sd = self.deviations.mean().values
    
    # Weights
    pred_weight = np.tile(self.weights.mean().values, (steps, 1))

    if 'weight' in predict:
      pred_weight = []

      for col in self.weights.columns[:-1]:
        weights_fit = FitData(method='logit')
        weights_fit.fit(self.time_periods, self.weights[col])
        weights_fit.predict(n_periods=steps)

        pred_weight.append( weights_fit.y_hat_robust[-steps:] )

      pred_weight = np.column_stack(pred_weight)
      pred_weight = np.column_stack(( pred_weight, 1 - pred_weight.sum(axis=1) ))

    self.pred_weight = pd.DataFrame(
        pred_weight,
        columns=self.weights.columns,
        index=range(self.T,self.T+steps)
    )

    self.new_mixtures = [
        GeneralMixtureModel([
            self.distributions(self.pred_mean.iloc[i,j], self.pred_sd[j]) for j in range(self.n_components)
            ], weights = self.pred_weight.iloc[i]) for i in range(steps)
        ]

  def sample(self, samples_per_period):
    return pd.DataFrame(
              np.row_stack(
                  [np.column_stack(
                      (self.new_mixtures[i].sample(samples_per_period), [i+self.T]*samples_per_period)
                      ) for i in range(self.steps)]),
              columns = self.vars + ['time']
    )

#%% PARAMETERS & CLASSES - FPCA
class Fpca():
  def __init__(self, data_matrix, grid_points, n_basis=None):
    self.data_matrix = data_matrix
    self.grid_points = grid_points
    self.time_periods = len(self.data_matrix)
    self.n_basis = n_basis
    
    self.fd = skfda.FDataGrid(self.data_matrix, self.grid_points)
    if self.n_basis:
      self.basis = skfda.representation.basis.BSpline(n_basis=self.n_basis, domain_range=self.fd.domain_range[0])
      self.fd = self.fd.to_basis(self.basis)

  def fit_fpca(self, n_components=2):
    self.n_components = n_components
    self.fpca = FPCA(n_components=self.n_components)
    self.fpca = self.fpca.fit(self.fd)
    self.components = np.squeeze(self.fpca.components_(self.grid_points))
    self.mean = self.fpca.mean_(self.grid_points).reshape(-1,1)
    self.scores = self.fpca.transform(self.fd)

    self.results = dict()
    for t in range(self.time_periods):
      out = np.tensordot( self.scores[t,:].reshape(1,-1), self.components, axes=1 ).reshape(-1,1) + self.mean
      self.results[t] = np.squeeze(out)

  def fit_time_series(self, models={}, h_steps=5):
    self.h_steps = h_steps
    if models:
      self.models = models
    else:
      self.models = {
          comp: auto_arima(self.scores[:,comp], start_p=0, d=1, start_q=0,
                          max_p=10, max_d=10, max_q=10, start_P=0,
                          D=1, start_Q=0, max_P=10, max_D=10,
                          max_Q=10, m=1, seasonal=False,
                          error_action='warn', trace=False,
                          stepwise=True, random_state=20, n_fits=1000 )
          for comp in range(self.n_components)
      }

    self.pred_w = np.column_stack((
        [self.models[comp].predict(n_periods=h_steps) for comp in range(self.n_components)]
    ))

  def predict(self, exp=True, normalize=True):
    self.pred_f = dict()

    for i in range(self.h_steps):
      self.pred_f[i] = np.squeeze( np.tensordot( self.pred_w[i,:].reshape(1,-1), self.components, axes=1 ).reshape(-1,1) + self.mean )

    if exp:
      self.pred_f = [np.exp(self.pred_f[i]) for i in range(self.h_steps)]

    if normalize:
      self.pred_f = [self.pred_f[i] / integrate.simpson(self.pred_f[i], self.grid_points) for i in range(self.h_steps)]

#%% PARAMETERS & CLASSES - WGAN
# Source: https://github.com/eriklindernoren/PyTorch-GAN/tree/master

class GeneratorModel(nn.Module):
  def __init__(self, latent_dim, hidden_layers, hidden_dim, input_dim, leaky_relu, device):
    super(GeneratorModel, self).__init__()
    
    self.device = device
    self.num_hidden_layers = hidden_layers

    self.hidden_layers = nn.ModuleList()

    self.hidden_layers.append(nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.LeakyReLU(leaky_relu)
        ))

    # Add the remaining hidden layers
    for i in range(self.num_hidden_layers - 1):
        self.hidden_layers.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu)
        ))

    self.output = nn.Sequential(
        nn.Linear(hidden_dim, input_dim)
    )
  
  def forward(self, x, labels):
    x = torch.column_stack((x, labels))

    out = x
    for hidden_layer in self.hidden_layers:
        out = hidden_layer(out)

    out = self.output(out)
    return out.to(self.device)

class DiscriminatorModel(nn.Module): 
  def __init__(self, input_dim, hidden_layers, hidden_dim, dropout, leaky_relu, device):
    super(DiscriminatorModel, self).__init__()
    
    self.device = device
    self.num_hidden_layers = hidden_layers

    self.hidden_layers = nn.ModuleList()

    self.hidden_layers.append(nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        ))
    
    # Add the remaining hidden layers
    for i in range(self.num_hidden_layers - 1):
        self.hidden_layers.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        ))

    self.output_layer = nn.Sequential(
      nn.Linear(hidden_dim, 1),
      nn.Sigmoid()
    )
    
  def forward(self, x, labels):
    x = torch.column_stack((x, labels))
    output = x

    for hidden_layer in self.hidden_layers:
        output = hidden_layer(output)

    output = self.output_layer(output)
    return output.to(self.device)

class Wgan():
  def __init__(self,
               data,
               vars,
               latent_dim,
               condition_dim,
               g_hidden_layers,
               g_hidden_dim,
               g_leaky_relu,
               d_hidden_layers,
               d_hidden_dim,
               d_dropout,
               d_leaky_relu,
               input_dim,
               lr_gen,
               lr_dis,
               b1,
               b2,
               lambda_gp,
               n_critic,
               batch_size,
               device):
    self.data = data
    self.vars = vars
    self.latent_dim = latent_dim
    self.condition_dim = condition_dim
    self.g_hidden_layers = g_hidden_layers
    self.g_hidden_dim = g_hidden_dim
    self.g_leaky_relu = g_leaky_relu
    self.d_hidden_layers = d_hidden_layers
    self.d_hidden_dim = d_hidden_dim
    self.d_dropout = d_dropout
    self.d_leaky_relu = d_leaky_relu
    self.input_dim = input_dim
    self.lr_gen = lr_gen
    self.lr_dis = lr_dis
    self.b1 = b1
    self.b2 = b2
    self.lambda_gp = lambda_gp
    self.n_critic = n_critic
    self.batch_size = batch_size
    self.device = device

    cuda = True if torch.cuda.is_available() else False
    self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    self.scaler = preprocessing.StandardScaler().fit(self.data[vars])
    self.time_scaler = preprocessing.MinMaxScaler().fit(self.data.time.values.reshape(-1,1))

    self.scaled = self.data.copy()
    self.scaled[vars] = self.scaler.transform(self.scaled[vars])
    self.scaled['time'] = self.time_scaler.transform(self.scaled.time.values.reshape(-1,1))
    self.scaled = self.scaled.to_numpy().astype(float)

    self.dataset = TensorDataset(torch.from_numpy(self.scaled).clone())
    self.loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True) # SHUFFLE False ?

    self.discriminator = DiscriminatorModel(self.input_dim, self.d_hidden_layers, self.d_hidden_dim, self.d_dropout, self.d_leaky_relu, self.device)
    self.generator = GeneratorModel(self.latent_dim, self.g_hidden_layers, self.g_hidden_dim, self.input_dim, self.g_leaky_relu, self.device)
    self.discriminator.to(self.device)
    self.generator.to(self.device)

    self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_dis, betas=(self.b1, self.b2))
    self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(self.b1, self.b2))

  def compute_gradient_penalty(self, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = self.Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = self.discriminator(interpolates, labels)
    fake = torch.autograd.Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

  def train(self, n_epochs, trace=True):
    sample_t = torch.tensor(self.data.time.astype(float))
   
    self.G_loss_total = []
    self.D_loss_total = []

    for epoch_idx in np.arange(n_epochs):
      G_loss = []
      D_loss = []

      for batch_idx, data_input in enumerate(self.loader):
        true_data = torch.autograd.Variable( data_input[0][:,:self.input_dim].to(self.device) )
        labels = data_input[0][:,-self.condition_dim]
        labels = labels.cpu().reshape(-1,1).to(self.device)
        
        self.discriminator_optimizer.zero_grad()

        noise = torch.autograd.Variable( torch.randn(true_data.size(0), self.latent_dim).to(self.device) )
        generated_data = self.generator(noise, labels.float())

        real_validity = self.discriminator( true_data.float(), labels.float() )
        fake_validity = self.discriminator( generated_data, labels.float() )
        gradient_penalty = self.compute_gradient_penalty( true_data.float(), generated_data, labels.float() )

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
        d_loss.backward()
        self.discriminator_optimizer.step()

        D_loss.append(d_loss.data.item())
        self.D_loss_total.append(d_loss.data.item())

        self.generator_optimizer.zero_grad()

        if batch_idx % self.n_critic == 0:
          generated_data = self.generator(noise, labels.float())
          fake_validity = self.discriminator( generated_data, labels.float() )
          g_loss = -torch.mean( fake_validity )

          g_loss.backward()
          self.generator_optimizer.step()

          G_loss.append(g_loss.data.item())
          self.G_loss_total.append(g_loss.data.item())

      
      if trace:  
          if epoch_idx % 10 == 0:
            #print(torch.mean(real_validity), torch.mean(fake_validity), lambda_gp * gradient_penalty)
            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                    (epoch_idx), n_epochs, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))
    
          if epoch_idx % 50 == 0:
            self.generator.eval()
            target = np.random.choice(sample_t)
            
            print('\n')
            print(target)
    
            noise = torch.autograd.Variable( torch.randn(self.batch_size, self.latent_dim).to(self.device) )
    
            t = self.time_scaler.transform(np.array([[target]]))[0][0]
            t = torch.tensor(t).repeat_interleave(self.batch_size).to(self.device)
            
            generated_data = self.generator(noise, t.float())
            generated_data = pd.DataFrame(generated_data.detach().cpu().numpy(), columns=self.vars)
            generated_data['obs'] = 'sample'
    
            samples = pd.DataFrame(self.scaled[self.data.time == target, :-self.condition_dim], columns=self.vars).copy()
            samples['obs'] = 'obs'
            
            samples = pd.concat([samples, generated_data], ignore_index=True)
            samples[self.vars] = self.scaler.inverse_transform( samples[self.vars] )
            
            try:
              sns.pairplot(samples, hue='obs')
              plt.show()
            except:
              pass
            self.generator.train()

  def sample(self, time_periods, sample_size=1000):
    self.generator.eval()
    noise = torch.autograd.Variable( torch.randn(sample_size, self.latent_dim).to(self.device) )

    samples = []

    for target in time_periods:
      t = self.time_scaler.transform(np.array([[target]]))[0][0]
      t = torch.tensor(t).repeat_interleave(sample_size).to(self.device)

      generated_data = self.generator(noise, t.float())      
      generated_data = pd.DataFrame(generated_data.detach().cpu().numpy(), columns=self.vars)
      generated_data['time'] = target
      generated_data[self.vars] = self.scaler.inverse_transform(generated_data[self.vars])

      samples.append(generated_data)
      
    samples = pd.concat(samples, ignore_index=True)

    return samples