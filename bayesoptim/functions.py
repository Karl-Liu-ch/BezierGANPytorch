import sys
sys.path.append('./')
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from model import Generator, loadmodel
import platform
from simulation import evaluate
import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-7
latent_dim = 3
noise_dim = 10
if platform.system().lower() == 'linux':
    path = '/work3/s212645/BezierGANPytorch/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
checkpoint_dir = path + 'ResNet_{}_{}_{}'.format(latent_dim, noise_dim, 256)

mass = 0.32
area = 0.2254
d = 0.155
thickness = 0.058

class Airfoil(object):
    
    def __init__(self):
        self.y = None
        self.bounds = None
        self.dim = None
            
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = - np.apply_along_axis(lambda x: evaluate(self.synthesize(x), modify_thickness=True)[-1], 1, x)
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
        generator = loadmodel(checkpoint_dir + '/generator.pth')
        self.model = generator
        self.latent_dim = 3
        self.noise_dim = 10
        latent_bounds = np.array([0., 1.])
        latent_bounds = np.tile(latent_bounds, [self.latent_dim, 1])
        noise_bounds = np.array([-0.5, 0.5])
        noise_bounds = np.tile(noise_bounds, [self.noise_dim, 1])
        self.bounds = np.vstack((latent_bounds, noise_bounds))
        bounds = (0.0, 1.0)
        y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(self.latent_dim))
        noise = np.random.normal(scale=0.5, size=(self.noise_dim))
        self.alpha0 = np.concatenate([y_latent, noise])

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.to(torch.float32)
        y_latent = x[:3].unsqueeze(dim=0)
        noise = x[3:].unsqueeze(dim=0)
        x_fake_train, _, _, _, _ = self.model(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        af = x_fake_train.reshape(256, 2).detach().cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        return af
     
class AirfoilP5B(Airfoil):
    def __init__(self, thickness = 0.058):
        super().__init__()
        self.thickness = thickness
        self.dim = 13
        generator = loadmodel(checkpoint_dir + '/generator.pth')
        self.model = generator
        self.latent_dim = 3
        self.noise_dim = 10
        latent_bounds = np.array([0., 1.])
        latent_bounds = np.tile(latent_bounds, [self.latent_dim, 1])
        noise_bounds = np.array([-0.5, 0.5])
        noise_bounds = np.tile(noise_bounds, [self.noise_dim, 1])
        self.bounds = np.vstack((latent_bounds, noise_bounds))
        bounds = (0.0, 1.0)
        y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(self.latent_dim))
        noise = np.random.normal(scale=0.5, size=(self.noise_dim))
        self.alpha0 = np.concatenate([y_latent, noise])

    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = np.apply_along_axis(lambda x: type2_simu(self.synthesize(x), mass=mass, diameter=d, area=area), 1, x)
        self.y = np.squeeze(y)
        return self.y
    
    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.to(torch.float32)
        y_latent = x[:3].unsqueeze(dim=0)
        noise = x[3:].unsqueeze(dim=0)
        x_fake_train, _, _, _, _ = self.model(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        af = x_fake_train.reshape(256, 2).detach().cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        return af
     
class AirfoilHickHenne(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 30
        self.bounds = np.array([[-1, 1]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        path = 'samples/DiffusionAirfoil1DTransform_001_-2.0_0.7F.dat'
        airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
        # airfoil = np.loadtxt(path, skiprows=1)
        airfoil = interpolate(airfoil, 256, 3)
        self.af = airfoil
        self.alpha0 = np.zeros([30])

    def synthesize(self, x):
        a_up0 = np.array([x[0] * 0.0001])
        a_up1 = x[1:6] * 0.01
        a_up2 = x[6:11] * 0.001
        a_up3 = x[11:15] * 0.0001
        a_up = np.concatenate([a_up0, a_up1, a_up2, a_up3])
        a_low0 = np.array([x[15] * 0.0001])
        a_low1 = x[16:21] * 0.01
        a_low2 = x[21:26] * 0.001
        a_low3 = x[26:] * 0.0001
        a_low = np.concatenate([a_low0, a_low1, a_low2, a_low3])
        af = np.copy(self.af)
        af = mute_airfoil(af, a_up=a_up, a_low=a_low)
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        return af
    
class AirfoilHickHenneP5B(Airfoil):
    def __init__(self, thickness = 0.058):
        super().__init__()
        self.thickness = thickness
        self.dim = 30
        self.bounds = np.array([[-1, 1]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        path = 'bayesoptim/bo_refine_2.dat'
        airfoil = np.loadtxt(path, skiprows=1)
        airfoil = interpolate(airfoil, 256, 3)
        self.af = airfoil
        self.alpha0 = np.zeros([30])
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = np.apply_along_axis(lambda x: type2_simu(self.synthesize(x), mass=mass, diameter=d, area=area), 1, x)
        self.y = np.squeeze(y)
        return self.y

    def synthesize(self, x):
        a_up0 = np.array([x[0] * 0.0001])
        a_up1 = x[1:6] * 0.01
        a_up2 = x[6:11] * 0.001
        a_up3 = x[11:15] * 0.0001
        a_up = np.concatenate([a_up0, a_up1, a_up2, a_up3])
        a_low0 = np.array([x[15] * 0.0001])
        a_low1 = x[16:21] * 0.01
        a_low2 = x[21:26] * 0.001
        a_low3 = x[26:] * 0.0001
        a_low = np.concatenate([a_low0, a_low1, a_low2, a_low3])
        af = np.copy(self.af)
        af = mute_airfoil(af, a_up=a_up, a_low=a_low)
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        return af
    
