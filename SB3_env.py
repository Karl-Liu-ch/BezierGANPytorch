from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import numpy as np
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
from simulation_win import *
from main import generator
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

def normalize_af(af):
    af[:,0] -= af[:,0].min()
    af /= (af[:,0].max() - af[:,0].min())
    return af

class OptimEnv(gym.Env):
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.06, maxsteps = 50, Re1 = 58000, Re2 = 400000, lamda = 5, alpha=0.2, mode = '2d'):
        self.cl = cl
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
        self.thickness = thickness
        self.action_space = spaces.Box(high=1., low=-1., shape=(1,13), dtype=np.float32)
        self.observation_space = spaces.Box(high=1., low=-1., shape=(1,512), dtype=np.float32)
        self.steps = 0
        self.maxsteps = maxsteps
        self.lamda = lamda
    
    def reset(self, seed=None, options=None):
        self.steps = 0
        successful = False
        while not successful:
            try:
                # self.airfoil = Diff1D_transform.sample(batch_size=1, channels=1).reshape(256, 2).cpu().numpy()
                # self.airfoil[:,1] = self.airfoil[:,1] * self.thickness / cal_thickness(self.airfoil)
                # self.airfoil[:,0] -= self.airfoil[:,0].min()
                # self.airfoil /= (self.airfoil[:,0].max() - self.airfoil[:,0].min())
                # self.airfoil = self.airfoil.reshape(1, 1, 512)
                # self.airfoil = torch.from_numpy(self.airfoil).to(device)

                self.airfoil = self.base_airfoil.reshape(1, 1, 512)
                self.state = self.airfoil.reshape(512)

                airfoil = self.airfoil.reshape(1, 256, 2)
                airfoil = airfoil.cpu().numpy()
                airfoil = airfoil[0]
                airfoil = derotate(airfoil)
                airfoil = normalize_af(airfoil)
                perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
                airfoil = setflap(airfoil, theta=-2)
                CD, _ = evalpreset(airfoil, Re=self.Re2)
                R = cd + CD * self.lamda
                if not np.isnan(R):
                    successful = True
                    print('Reset Successful: CL/CD={:.4f}, R={}'.format(perf, R))
            except Exception as e:
                print(e)
        self.R_prev = R
        self.Rbl = R
        self.R = R
        info = {}
        return self.state.reshape(1,512).cpu().numpy(), info
    
    def step(self, action):
        self.steps += 1
        y_latent = (torch.from_numpy(action.reshape([13])[:3]).reshape([1,3]).to(device) + 1.0) / 2.0
        noise = torch.from_numpy(action.reshape([13])[3:]).to(device).reshape([1,10])
        x_fake_train, _, _, _, _ = generator(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        af = x_fake_train.reshape(256, 2).detach().cpu().numpy()
        af[:,0] -= af[:,0].min()
        af /= (af[:,0].max() - af[:,0].min())
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        af = torch.from_numpy(af).to(device)
        af = af.reshape([1,1,512]).detach()
        
        self.airfoil = af  * self.alpha + (1-self.alpha) * self.airfoil
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        successful = False
        try:
            perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
            airfoil = setflap(airfoil, theta=-2)
            CD, _ = evalpreset(airfoil, Re=self.Re2)
            R = cd + CD * self.lamda
            print('Successful: CL/CD={:.4f}, R={}'.format(perf, R))
            # perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=self.lamda, check_thickness=False)
            successful = True
        except:
            successful = False
            R = np.nan
        if np.isnan(R):
            reward = -1
            reward_final = -1
        else:
            reward_final = (self.Rbl - R) * 100
            reward = 0.01 / R
            self.R_prev = R
        # print(reward)
        if R < self.R:
            self.R = R
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        self.state = self.airfoil.reshape(512)
        
        truncated = False
        done = False
        if R < 0.0166 + 0.004852138459682465 * self.lamda and perf > 40:
            done = True
            reward += 100
            truncated = False
        if self.steps > self.maxsteps:
            done = True
            truncated = True
            reward += reward_final
        reward_final = {'reward_final': reward_final}
        if not successful:
            reward = -10
            done = True
            truncated = True
        return self.state.reshape(1,512).detach().cpu().numpy(), reward, done, truncated, reward_final


class AirfoilEnv(gym.Env):
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.06, maxsteps = 50, Re1 = 58000, Re2 = 400000, lamda = 5, alpha=0.2, mode = '2d', use_xfoil = False):
        self.cl = cl
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
        self.thickness = thickness
        self.action_space = spaces.Box(high=1., low=-1., shape=(1,13), dtype=np.float32)
        self.observation_space = spaces.Box(high=1., low=-1., shape=(1,512), dtype=np.float32)
        self.steps = 0
        self.maxsteps = maxsteps
        self.lamda = lamda
        self.use_xfoil = use_xfoil
    
    def reset(self, seed=None, options=None):
        self.steps = 0
        successful = False
        while not successful:
            try:
                self.airfoil = self.base_airfoil.reshape(1, 1, 512)
                self.state = self.airfoil.reshape(512)

                airfoil = self.airfoil.reshape(1, 256, 2)
                airfoil = airfoil.cpu().numpy()
                airfoil = airfoil[0]
                airfoil = derotate(airfoil)
                airfoil = normalize_af(airfoil)
                if self.use_xfoil:
                    perf, cd = evalperf_win(airfoil, cl = self.cl, Re = self.Re1)
                else:
                    perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
                # airfoil = setflap(airfoil, theta=-2)
                # CD, _ = evalpreset(airfoil, Re=self.Re2)
                # R = cd + CD * self.lamda
                if not np.isnan(perf):
                    successful = True
                    print('Reset Successful: CL/CD={:.4f}'.format(perf))
            except Exception as e:
                print(e)
        self.perf_prev = perf
        self.perfbl = perf
        self.perf = perf
        info = {}
        af = np.copy(airfoil)
        af[:,0] = af[:,0] * 2.0 - 1.0
        af[:,1] = af[:,1] * 10.0
        self.state = torch.from_numpy(af).to(device) 
        return self.state.reshape(1,512).cpu().numpy(), info
    
    def step(self, action):
        self.steps += 1
        y_latent = (torch.from_numpy(action.reshape([13])[:3]).reshape([1,3]).to(device) + 1.0) / 2.0
        noise = torch.from_numpy(action.reshape([13])[3:]).to(device).reshape([1,10])
        x_fake_train, _, _, _, _ = generator(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        af = x_fake_train.reshape(256, 2).detach().cpu().numpy()
        af[:,0] -= af[:,0].min()
        af /= (af[:,0].max() - af[:,0].min())
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        af = torch.from_numpy(af).to(device)
        af = af.reshape([1,1,512]).detach()
        
        self.airfoil = af  * self.alpha + (1-self.alpha) * self.airfoil
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        successful = False
        try:
            if self.use_xfoil:
                perf, cd = evalperf_win(airfoil, cl = self.cl, Re = self.Re1)
            else:
                perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
            # airfoil = setflap(airfoil, theta=-2)
            # CD, _ = evalpreset(airfoil, Re=self.Re2)
            # R = cd + CD * self.lamda
            print('Successful: CL/CD={:.4f}'.format(perf))
            # perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=self.lamda, check_thickness=False)
            successful = True
        except:
            successful = False
            perf = np.nan
        reward = 0
        reward_final = 0
        if np.isnan(perf):
            reward = -0.1
            reward_final = -0.1
        else:
            reward_final = (perf / self.perfbl) * 10
            reward = (perf / 39.0) ** 10 / 2.0
            self.perf_prev = perf
        print(reward)
        if perf > self.perf:
            self.perf = perf
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        af = np.copy(airfoil)
        af[:,0] = af[:,0] * 2.0 - 1.0
        af[:,1] = af[:,1] * 10.0
        self.state = torch.from_numpy(af).to(device) 
        # self.state = self.airfoil.reshape(512)
        
        truncated = False
        done = False
        if perf > 41:
            done = True
            reward += self.maxsteps + 10 - self.steps
            truncated = False
        if self.steps > self.maxsteps:
            done = True
            truncated = True
            reward += reward_final
        if not successful:
            # reward = -10
            done = True
            truncated = True
        reward_final = {'reward_final': reward_final}
        return self.state.reshape(1,512).detach().cpu().numpy(), reward, done, truncated, reward_final

class HicksHenneEnv(AirfoilEnv):
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.06, maxsteps = 50, Re1 = 58000, Re2 = 400000, lamda = 5, alpha=0.2, mode = '2d', use_xfoil = False):
        self.cl = cl
        self.base_airfoil = base_airfoil
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
        self.thickness = thickness
        self.action_space = spaces.Box(high=1., low=-1., shape=(1,30), dtype=np.float32)
        self.observation_space = spaces.Box(high=1., low=-1., shape=(1,512), dtype=np.float32)
        self.steps = 0
        self.maxsteps = maxsteps
        self.lamda = lamda
        self.use_xfoil = use_xfoil
    
    def reset(self, seed=None, options=None):
        self.steps = 0
        successful = False
        while not successful:
            try:
                self.airfoil = np.copy(self.base_airfoil)
                airfoil = self.airfoil
                airfoil = derotate(airfoil)
                airfoil = normalize_af(airfoil)
                if self.use_xfoil:
                    perf, cd = evalperf_win(airfoil, cl = self.cl, Re = self.Re1)
                else:
                    perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
                # airfoil = setflap(airfoil, theta=-2)
                # CD, _ = evalpreset(airfoil, Re=self.Re2)
                # R = cd + CD * self.lamda
                if not np.isnan(perf):
                    successful = True
                    print('Reset Successful: CL/CD={:.4f}'.format(perf))
            except Exception as e:
                print(e)
        self.perf_prev = perf
        self.perfbl = perf
        self.perf = perf
        info = {}
        af = np.copy(airfoil)
        af[:,0] = af[:,0] * 2.0 - 1.0
        af[:,1] = af[:,1] / self.thickness
        self.state = torch.from_numpy(af).to(device) 
        return self.state.reshape(1,512).cpu().numpy(), info
    
    def step(self, action):
        self.steps += 1
        x = action.reshape([30])
        a_up0 = x[0:1] * 0.0001
        a_up1 = x[1:6] * 0.001
        a_up2 = x[6:11] * 0.001
        a_up3 = x[11:15] * 0.0001
        a_up = np.concatenate([a_up0, a_up1, a_up2, a_up3])
        a_low0 = x[15:16] * 0.0001
        a_low1 = x[16:21] * 0.001
        a_low2 = x[21:26] * 0.001
        a_low3 = x[26:] * 0.0001
        a_low = np.concatenate([a_low0, a_low1, a_low2, a_low3])
        af = np.copy(self.airfoil)
        af = mute_airfoil(af, a_up=a_up, a_low=a_low)
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        
        airfoil = af
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        self.airfoil = np.copy(airfoil)
        successful = False
        try:
            if self.use_xfoil:
                perf, cd = evalperf_win(airfoil, cl = self.cl, Re = self.Re1)
            else:
                perf, _, cd = evalperf(airfoil, cl = self.cl, Re = self.Re1)
            # airfoil = setflap(airfoil, theta=-2)
            # CD, _ = evalpreset(airfoil, Re=self.Re2)
            # R = cd + CD * self.lamda
            print('Successful: CL/CD={:.4f}'.format(perf))
            # perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=self.lamda, check_thickness=False, modify_thickness=True)
            successful = True
        except:
            successful = False
            perf = np.nan
        reward = 0
        reward_final = 0
        if np.isnan(perf):
            reward = -0.1
            reward_final = -0.1
        else:
            reward_final = (perf / self.perfbl) * 10
            reward = (perf / 39.0) ** 10 / 2.0
            self.perf_prev = perf
        print(reward)
        if perf > self.perf:
            self.perf = perf
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        af = np.copy(airfoil)
        af[:,0] = af[:,0] * 2.0 - 1.0
        af[:,1] = af[:,1] / self.thickness
        self.state = torch.from_numpy(af).to(device) 
        # self.state = self.airfoil.reshape(512)
        
        truncated = False
        done = False
        if perf > 41:
            done = True
            reward += self.maxsteps + 10 - self.steps
            truncated = False
        if self.steps > self.maxsteps:
            done = True
            truncated = True
            reward += reward_final
        if not successful:
            # reward = -10
            done = True
            truncated = True
        reward_final = {'reward_final': reward_final}
        return self.state.reshape(1,512).detach().cpu().numpy(), reward, done, truncated, reward_final

if __name__ == '__main__':
    env = OptimEnv()
    check_env(env)