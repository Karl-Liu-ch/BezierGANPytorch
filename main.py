import os
import torch
import torch.nn as nn
from model import Generator, Discriminator
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import platform
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# torch.manual_seed(7) # cpu
# # torch.cuda.manual_seed(7) #gpu
# np.random.seed(7) #numpy
# random.seed(7) #random and transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-7
latent_dim = 3
noise_dim = 10

def preprocess(X):
    X = np.expand_dims(X, axis=-1)
    return X.astype(np.float32)

def postprocess(X):
    X = np.squeeze(X)
    return X

def Normalize(airfoil):
    r = np.maximum(airfoil[0,0], airfoil[-1,0])
    r = float(1.0/r)
    return airfoil * r

def Save_model(generator, discrimiator, checkpoint_dir):
    torch.save(generator.state_dict(), checkpoint_dir + '/generator.pth')
    torch.save(discrimiator.state_dict(), checkpoint_dir + '/discrimiator.pth')

def train(X_train, path, latent_dim = 3, noise_dim = 10, train_steps = 2000, batch_size = 64, save_interval = 0, director = '.', load_models = False):
    bounds = (0.0, 1.0)
    X_train = preprocess(X_train)
    # ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
    generator = Generator(latent_dim=latent_dim, noise_dim=noise_dim, n_points=X_train.shape[1]).to(device)
    discriminator = Discriminator(latent_dim=latent_dim, n_points=X_train.shape[1]).to(device)
    checkpoint_dir = path + "ResNet_{}_{}_{}".format(latent_dim, noise_dim, X_train.shape[1])
    try:
        os.mkdir(checkpoint_dir)
    except:
        pass
    try:
        generator_path = checkpoint_dir + '/generator.pth'
        discriminator_path = checkpoint_dir + '/discrimiator.pth'
        state_dict = torch.load(generator_path)
        generator.load_state_dict(state_dict)
        state_dict = torch.load(discriminator_path)
        discriminator.load_state_dict(state_dict)
        print("load models successfully")
    except:
        pass
    X_train = torch.from_numpy(X_train).to(device)
    X_train = TensorDataset(X_train, X_train)
    X_train = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    learning_rate = 3e-4
    optim_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    schedulerg = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, train_steps, eta_min=1e-6)
    schedulerd = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, train_steps, eta_min=1e-6)
    epoch = 0
    while epoch < train_steps:
        for i, (X_real, label) in enumerate(X_train):
            # train discriminator:
            # train d_real
            y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, noise_dim))
            y_latent = torch.from_numpy(y_latent).to(device)
            y_latent = y_latent.float()
            noise = torch.from_numpy(noise).to(device)
            noise = noise.float()
            d_real, _ = discriminator(X_real)
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            d_loss_real = torch.mean(BCEWithLogitsLoss(d_real, torch.ones_like(d_real)))

            # train d_fake
            x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
            d_fake, q_fake_train = discriminator(x_fake_train.detach())
            d_loss_fake = torch.mean(BCEWithLogitsLoss(d_fake, torch.zeros_like(d_fake)))
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            q_target = y_latent
            epsilon = (q_target - q_mean) / (torch.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * torch.square(epsilon)
            q_loss = torch.mean(q_loss)
            d_train_real_loss = d_loss_real
            d_train_fake_loss = d_loss_fake + q_loss
            optim_d.zero_grad()
            d_train_real_loss.backward()
            d_train_fake_loss.backward()
            optim_d.step()
            # print("training Discriminator. D real loss:", d_train_real_loss.item(), "D fake loss:", d_train_fake_loss.item())

            # train g_loss
            y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, noise_dim))
            y_latent = torch.from_numpy(y_latent).to(device)
            y_latent = y_latent.float()
            noise = torch.from_numpy(noise).to(device)
            noise = noise.float()
            x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
            d_fake, q_fake_train = discriminator(x_fake_train)
            g_loss = torch.mean(BCEWithLogitsLoss(d_fake, torch.ones_like(d_fake)))

            # Regularization for w, cp, a, and b
            r_w_loss = torch.mean(w_train[:,1:-1], dim=(1,2))
            cp_dist = torch.norm(cp_train[:, 1:] - cp_train[:, :-1], dim=-1)
            r_cp_loss = torch.mean(cp_dist, dim=-1)
            r_cp_loss1 = torch.max(cp_dist, dim=-1)[0]
            ends = cp_train[:, 0] - cp_train[:, -1]
            r_ends_loss = torch.norm(ends, dim=-1) + torch.maximum(torch.tensor(0.0).to(device), -10 * ends[:, 1])
            r_db_loss = torch.mean(db_train * torch.log(db_train), dim=-1)
            r_loss = r_w_loss + r_cp_loss + 0 * r_cp_loss1 + r_ends_loss + 0 * r_db_loss
            r_loss = torch.mean(r_loss)
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            q_target = y_latent
            epsilon = (q_target - q_mean) / (torch.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * torch.square(epsilon)
            q_loss = torch.mean(q_loss)
            # Gaussian loss for Q
            G_loss = g_loss + 10*r_loss + q_loss
            optim_g.zero_grad()
            G_loss.backward()
            optim_g.step()
            # print("traning Generator. G loss: ", G_loss.item())
        epoch += 1
        if (epoch % 10) == 0:
            print("saving model, epoch:", epoch)
            Save_model(generator, discriminator, checkpoint_dir)

        schedulerg.step()
        schedulerd.step()
        print("epoch: ", epoch, "G loss: ", G_loss.item(), "D real loss: ", d_train_real_loss.item(), "D fake loss: ", d_train_fake_loss.item())

def eval(model_path, latent_dim = 3, noise_dim = 10, n_points = 256):
    print(model_path)
    generator = Generator(latent_dim=latent_dim, noise_dim=noise_dim, n_points=n_points).to(device)
    state_dict = torch.load(model_path)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

def sample(generator, batch_size):
    bounds = (0.0, 1.0)
    y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, latent_dim))
    noise = np.random.normal(scale=0.5, size=(batch_size, noise_dim))
    y_latent = torch.from_numpy(y_latent).to(device)
    y_latent = y_latent.float()
    noise = torch.from_numpy(noise).to(device)
    noise = noise.float()
    x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
    x_fake_train = x_fake_train.squeeze(dim=-1)
    airfoil = x_fake_train.detach().cpu().numpy()
    return airfoil

if __name__ == '__main__':
    data = np.load('data/airfoil_interp.npy')
    if platform.system().lower() == 'linux':
        try:
            os.mkdir('/work3/s212645/BezierGANPytorch/')
        except:
            pass
    if platform.system().lower() == 'linux':
        path = '/work3/s212645/BezierGANPytorch/checkpoint/'
    elif platform.system().lower() == 'windows':
        path = 'H:/深度学习/checkpoint/'
    try:
        os.mkdir(path)
    except:
        pass
    # train(data, path)
    
    checkpoint_dir = path + "ResNet_{}_{}_{}".format(latent_dim, noise_dim, 256)
    generator = eval(checkpoint_dir + '/generator.pth')
    
    B = 2 ** 8
        
    airfoil = sample(generator, batch_size=B)[0]
    fig, axs = plt.subplots(1, 1)
    axs.plot(airfoil[:,0], airfoil[:,1])
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig('sample.png')
    plt.close()
    
    if platform.system().lower() == 'linux':
        path = '/work3/s212645/BezierGANPytorch/Airfoils/'
    elif platform.system().lower() == 'windows':
        path = 'H:/深度学习/AirfoilsSamples/'
    try:
        os.mkdir(path)
    except:
        pass
        
    for i in range(1000):
        num = str(i+1000).zfill(3)
        airfoil = sample(generator, batch_size=256)
        np.save(path + num + '.npy', airfoil)
        print(num + ' saved')