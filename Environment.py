import torch
import numpy as np
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
from main import Normalize
from scipy.signal import savgol_filter
from main import *
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-7
latent_dim = 3
noise_dim = 10

try:
    os.mkdir('checkpoint')
except:
    pass
bounds = (0.0, 1.0)
data = np.load('data/airfoil_interp.npy')
checkpoint_dir = "/work3/s212645/BezierGANPytorch/checkpoint/ResNet_{}_{}_{}".format(latent_dim, noise_dim, 256)
generator = eval(checkpoint_dir + '/generator.pth')
y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(1, latent_dim))
noise = np.random.normal(scale=0.5, size=(1, noise_dim))
y_latent = torch.from_numpy(y_latent).to(device)
y_latent = y_latent.float()
noise = torch.from_numpy(noise).to(device)
noise = noise.float()
x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)

def sample(generator, y_noise):
    noise = y_noise[:,:noise_dim]
    y_latent = y_noise[:,noise_dim:]
    x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
    return x_fake_train

base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

class OptimEnv():
    def __init__(self):
        self.cl = 0.65
        self.R = 1
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = 0.01
    
    def reset(self):
        y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(1, latent_dim))
        noise = np.random.normal(scale=0.5, size=(1, noise_dim))
        y_latent = torch.from_numpy(y_latent).to(device)
        y_latent = y_latent.float()
        noise = torch.from_numpy(noise).to(device)
        noise = noise.float()
        
        self.noise = torch.concat([noise, y_latent], dim=-1)
        self.airfoil = sample(generator, y_noise = self.noise)
        self.airfoil = self.airfoil.reshape(1, 256, 2) * self.alpha + (1-self.alpha) * self.base_airfoil.reshape(1, 256, 2)
        self.state = self.airfoil.reshape(512)
        return self.state.detach().cpu().numpy()
    
    def step(self, action):
        y_latent = (torch.from_numpy(action[:3]).reshape([1,3]).to(device) + 1.0) / 2.0
        noise = torch.from_numpy(action[3:]).to(device).reshape([1,10])
        self.noise = torch.concat([noise, y_latent], dim=-1)
        self.airfoil = sample(generator, y_noise = self.noise).reshape(256, 2) * self.alpha + (1-self.alpha) * self.airfoil.reshape(256, 2)
        airfoil = self.airfoil.reshape(256, 2)
        airfoil = airfoil.detach().cpu().numpy()
        thickness = cal_thickness(airfoil)
        perf, CD, af, R = evaluate(airfoil, self.cl, lamda=5, check_thickness=False)
        # print(f'perf: {perf}, R: {R}')
        if np.isnan(R):
            reward = -1
        else:
            reward = (0.042 - R) * 10 + thickness - 0.058
        print(reward)
        if R < self.R:
            self.R = R
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        self.state = self.airfoil.reshape(512)
        
        if perf > 50:
            done = True
            reward += 100
        else:
            done = False
        info = None
        return self.state.detach().cpu().numpy(), reward, done, info