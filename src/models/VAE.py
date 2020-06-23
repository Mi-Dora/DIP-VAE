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
        """
        Since the network is a fully connected network,
        it does not have the property like CNN that could receive image in multi-dimensions.
        Using dynamic image size and channels as parameters,
        so the network could handle different datasets
        :param img_size: (int) size of the input data, which should be square
        :param channel: (int) number of channels of the input data
        """
        super(VAE, self).__init__()
        self.img_size = img_size
        self.channel = channel

        # fully connected layers for the encoder
        # Key point to make the origin VAE from pytorch/examples able to handle RGB image with different size
        self.fc1 = nn.Linear(self.img_size**2*self.channel, 1600)
        self.fc11 = nn.Linear(1600, 400)
        self.fc21 = nn.Linear(400, 40)
        self.fc22 = nn.Linear(400, 40)
        # latent vector may be enlarged to have stronger expressing capability (currently 20->?)
        # so is the number of hidden layers

        # fully connected layers for the decoder
        self.fc3 = nn.Linear(40, 400)
        self.fc31 = nn.Linear(400, 1600)
        self.fc4 = nn.Linear(1600, self.img_size**2*self.channel)

    def encode(self, x):
        """
        :param x: (tensor) batch_size x image_size^2 x channel (3 for RGB)
        :return: (tensor, tensor) N(Jx1 mu, Jx1 Sigma)
        """
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc11(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # e^(0.5*ln(sigma^2)) = sigma = std
        std = torch.exp(0.5 * logvar)
        # get random numbers from a normal distribution with mean 0 and variance 1
        eps = torch.randn_like(std)
        # refer to Equation(17) on my report
        return mu + eps * std

    def decode(self, z):
        """
        :param z: (tensor) J x 1 latent variable
        :return: (tensor) reconstructed image
        """
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc31(h3))
        # sigmoid activation function to ensure the output in [0, 1]
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Forward propagation
        :param x: (tensor) batch_size x C x H x W
        :return: (tensor, tensor, tensor) reconstructed image, N(Jx1 mu, Jx1 Sigma)
        """
        mu, logvar = self.encode(x.view(-1, self.img_size**2*self.channel))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

