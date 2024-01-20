import sys
sys.path.append('./')
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from main import sample, generator
import platform
from simulation import evaluate
import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-7
latent_dim = 3
noise_dim = 10

class Airfoil(object):
    
    def __init__(self):
        self.y = None
        self.bounds = None
        self.dim = None
            
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = - np.apply_along_axis(lambda x: evaluate(self.synthesize(x))[-1], 1, x)
        self.y = np.squeeze(y)
        return self.y
    
    def is_feasible(self, x):
        x = np.array(x, ndmin=2)
        if self.y is None:
            self.y = self.__call__(x)
        feasibility = np.logical_not(np.isnan(self.y))
        return feasibility
    
    def synthesize(self, x):
        pass
    
    def sample_design_variables(self, n_sample, method='random'):
        if method == 'lhs':
            x = lhs(self.dim, samples=n_sample, criterion='cm')
            x = x * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            # x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(n_sample, self.dim))
            x = np.random.normal(size=(n_sample, self.dim))
        return np.squeeze(x)
    
    def sample_airfoil(self, n_sample, method='random'):
        x = self.sample_design_variables(n_sample, method)
        airfoils = self.synthesize(x)
        return airfoils
    
class AirfoilDiffusion(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 13
        self.model = generator
        self.latent_dim = 3
        self.noise_dim = 10
        latent_bounds = np.array([0., 1.])
        latent_bounds = np.tile(latent_bounds, [self.latent_dim, 1])
        noise_bounds = np.array([-0.5, 0.5])
        noise_bounds = np.tile(noise_bounds, [self.noise_dim, 1])
        self.bounds = np.vstack((latent_bounds, noise_bounds))

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.to(torch.float32)
        y_latent = x[:3].unsqueeze(dim=0)
        noise = x[3:].unsqueeze(dim=0)
        x_fake_train, _, _, _, _ = generator(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        af = x_fake_train.reshape(256, 2).detach().cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        return af