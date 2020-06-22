# -*- coding: utf-8 -*-
"""
    Created on Friday, Jun 19 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jun 23 2020

South East University Automation College, 211189 Nanjing China
"""
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, img_size, channel):
        super(VAE, self).__init__()
        self.img_size = img_size
        self.channel = channel
        self.fc1 = nn.Linear(self.img_size**2*self.channel, 800)
        self.fc21 = nn.Linear(800, 40)
        self.fc22 = nn.Linear(800, 40)

        # latent vector may be enlarged to have stronger expressing ability (20->?）
        self.fc3 = nn.Linear(40, 800)
        self.fc4 = nn.Linear(800, self.img_size**2*self.channel)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.img_size**2*self.channel))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

