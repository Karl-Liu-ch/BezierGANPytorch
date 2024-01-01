import os
import torch
import torch.nn as nn
from model import Generator, Discriminator
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

def train(X_train, latent_dim = 3, noise_dim = 10, train_steps = 2000, batch_size = 64, save_interval = 0, director = '.', load_models = False):
    bounds = (0.0, 1.0)
    X_train = preprocess(X_train)
    # ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
    generator = Generator(latent_dim=latent_dim, noise_dim=noise_dim, n_points=X_train.shape[1]).to(device)
    discriminator = Discriminator(latent_dim=latent_dim, n_points=X_train.shape[1]).to(device)
    checkpoint_dir = "checkpoint/ResNet_{}_{}_{}".format(latent_dim, noise_dim, X_train.shape[1])
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
    for epoch in range(train_steps):
        learning_rate = 1e-4
        # if epoch > 100:
        #     learning_rate = 1e-5
        # if epoch > 1000:
        #     learning_rate = 1e-6
        optim_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
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

        if (epoch % 500) == 0:
            print("saving model, epoch:", epoch)
            Save_model(generator, discriminator, checkpoint_dir)

        print("epoch: ", epoch, "G loss: ", G_loss.item(), "D real loss: ", d_train_real_loss.item(), "D fake loss: ", d_train_fake_loss.item())

def eval(model_path, latent_dim = 3, noise_dim = 10, n_points = 256):
    generator = Generator(latent_dim=latent_dim, noise_dim=noise_dim, n_points=n_points).to(device)
    state_dict = torch.load(model_path)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


if __name__ == '__main__':
    try:
        os.mkdir('checkpoint')
    except:
        pass
    for i in range(1000):
        num = str(i).zfill(3)
        B = 256
        bounds = (0.0, 1.0)
        data = np.load('data/airfoil_interp.npy')
        checkpoint_dir = "checkpoint/ResNET_{}_{}_{}".format(latent_dim, noise_dim, 256)
        generator = eval(checkpoint_dir + '/generator.pth')
        y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(B, latent_dim))
        noise = np.random.normal(scale=0.5, size=(B, noise_dim))
        y_latent = torch.from_numpy(y_latent).to(device)
        y_latent = y_latent.float()
        noise = torch.from_numpy(noise).to(device)
        noise = noise.float()
        x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
        x_fake_train = x_fake_train.squeeze(dim=-1)
        airfoil = x_fake_train.detach().cpu().numpy()
        np.save('sample/' + num + '.npy', airfoil)
    
    # fig, axs = plt.subplots(1, 1)
    # axs.plot(airfoil[:,0], airfoil[:,1])
    # axs.set_aspect('equal', 'box')
    # fig.tight_layout()
    # plt.savefig('sample.png')
    # plt.close()

    # train(data)