import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Dense1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU()
        )
        self.Dense2 = nn.Sequential(
            nn.Linear(256, 3),
            nn.Softmax()
        )

    def forward(self, input):
        feed_forward = self.Dense1(input)
        logits = self.Dense2(feed_forward)
        return logits


class Agent:
    LEARNING_RATE = 1e-4
    LAYER_SIZE = 256
    GAMMA = 0.9
    OUTPUT_SIZE = 3

    def __init__(self, state_size, window_size, trend, skip):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        # self.X = tf.placeholder(tf.float32, (None, self.state_size))
        # self.REWARDS = tf.placeholder(tf.float32, (None))
        # self.ACTIONS = tf.placeholder(tf.int32, (None))
        # feed_forward = tf.layers.dense(self.X, self.LAYER_SIZE, activation=tf.nn.relu)
        # self.logits = tf.layers.dense(feed_forward, self.OUTPUT_SIZE, activation=tf.nn.softmax)
        # input_y = tf.one_hot(self.ACTIONS, self.OUTPUT_SIZE)
        # loglike = tf.log((input_y * (input_y - self.logits) + (1 - input_y) * (input_y + self.logits)) + 1)
        # rewards = tf.tile(tf.reshape(self.REWARDS, (-1, 1)), [1, self.OUTPUT_SIZE])
        # self.cost = -tf.reduce_mean(loglike * (rewards + 1))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.cost)




if __name__ == '__main__':
    df = pd.read_csv('dataset/GOOG-year.csv')
    print(df.head())