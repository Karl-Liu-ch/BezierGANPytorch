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
checkpoint_dir = "checkpoint/ResNET_{}_{}_{}".format(latent_dim, noise_dim, 256)
generator = eval(checkpoint_dir + '/generator.pth')
y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(1, latent_dim))
noise = np.random.normal(scale=0.5, size=(1, noise_dim))
y_latent = torch.from_numpy(y_latent).to(device)
y_latent = y_latent.float()
noise = torch.from_numpy(noise).to(device)
noise = noise.float()
x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)

def derotate(airfoil):
    ptail = 0.5 * (airfoil[0,:]+airfoil[-1,:])
    ptails = np.expand_dims(ptail, axis=0)
    ptails = np.repeat(ptails, 256, axis=0)
    i = np.linalg.norm(airfoil - ptails, axis=1).argmax()
    phead = airfoil[i,:]
    theta = np.arctan2(-(airfoil[i,1] - ptail[1]), -(airfoil[i,0] - ptail[0]))
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    airfoil_R = airfoil
    airfoil_R -= np.repeat(np.expand_dims(phead, axis=0), 256, axis=0)
    airfoil_R = np.matmul(airfoil_R, R)
    return airfoil_R

def sample(generator, y_noise):
    noise = y_noise[:,:noise_dim]
    y_latent = y_noise[:,noise_dim:]
    x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
    return x_fake_train

class OptimEnv():
    def __init__(self):
        self.cl = 0.65
        self.best_perf = 0
    
    def reset(self):
        y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(1, latent_dim))
        noise = np.random.normal(scale=0.5, size=(1, noise_dim))
        y_latent = torch.from_numpy(y_latent).to(device)
        y_latent = y_latent.float()
        noise = torch.from_numpy(noise).to(device)
        noise = noise.float()
        
        self.noise = torch.concat([noise, y_latent], dim=-1)
        self.airfoil = sample(generator, y_noise = self.noise)
        self.state = torch.concat([self.noise, self.airfoil.reshape(1, 512)], dim=-1).squeeze(dim=1)
        return self.state.detach().cpu().numpy()
    
    def step(self, action):
        self.noise += torch.from_numpy(action).reshape([1,13]).to(device)
        self.airfoil = sample(generator, y_noise = self.noise)
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.detach().cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
        airfoil[:,0] = xhat
        airfoil[:,1] = yhat
        perf = evaluate(airfoil, self.cl)
        print(perf)
        if perf == np.nan:
            reward = 0
        else:
            reward = perf * 0.01
        if perf > self.best_perf:
            self.best_perf = perf
            np.savetxt('results/airfoilPPO.dat', airfoil)
        self.state = torch.concat([self.noise, self.airfoil.reshape(1, 512)], dim=-1).squeeze(dim=1)
        
        if perf > 50:
            done = True
            reward += 100
        else:
            done = False
        info = None
        return self.state.detach().cpu().numpy(), reward, done, info