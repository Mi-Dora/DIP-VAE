# -*- coding: utf-8 -*-
"""
    Created on Friday, Jun 19 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jun 23 2020

South East University Automation College, 211189 Nanjing China
"""
from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from src.models import VAE
from src.datasets import CustomDataset


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_size**2 * channel), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    torch.save(model.state_dict(), 'weights/{0}_{1}.pkl'.format(dataset_type, epoch))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    with open('logs/{0}_log.txt'.format(dataset_type), 'a') as logger:
        logger.write(
            "%d\t%f\t"
            % (epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    global best_loss
    global best_epoch
    comparison = None
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, -1, img_size, img_size)[:n]])
                save_image(comparison,
                           reconstruct_path + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    with open('logs/{0}_log.txt'.format(dataset_type), 'a') as logger:
        logger.write("%f\n" % test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        try:
            os.remove('weights/best_{0}_{1}.pkl'.format(dataset_type, best_epoch))
            os.remove(reconstruct_path + 'best_' + str(best_epoch) + '.png')
            os.remove(sample_path + 'best_' + str(best_epoch) + '.png')
        except FileNotFoundError:
            pass
        torch.save(model.state_dict(), 'weights/best_{0}_{1}.pkl'.format(dataset_type, epoch))
        if comparison is not None:
            save_image(comparison,
                       reconstruct_path + 'best_' + str(epoch) + '.png', nrow=n)
        best_epoch = epoch
        return True
    else:
        return False


if __name__ == "__main__":

    batch_size = 128
    epochs = 500
    seed = 1  # random seed
    log_interval = 10  # how many batches to wait before logging training status

    img_size = 32  # size for data image (square)

    sample_path = 'results/samples/'
    reconstruct_path = 'results/reconstructions/'
    weight_path = 'weights/'
    logger_path = 'logs/'

    dataset_path = 'D:/Mi-Dora/Source/IntroductionAI/data/img_align_celeba'
    # dataset_path = 'D:/Mi-Dora/Source/IntroductionAI/data/mnist'

    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(reconstruct_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(logger_path, exist_ok=True)

    dataset_type = dataset_path.split('/')[-1]

    best_loss = np.float('inf')
    best_epoch = -1

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if dataset_type == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_path, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_path, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        channel = 1
    else:
        # Using DataLoader to apply mini-batch and shuffle function
        train_loader = torch.utils.data.DataLoader(
            CustomDataset(
                dataset_path,
                train=True,
                img_size=img_size,
                in_memory=False
            ),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            CustomDataset(
                dataset_path,
                train=False,
                img_size=img_size,
                in_memory=False
            ),
            batch_size=batch_size,
            shuffle=True
        )
        channel = 3

    model = VAE(img_size=img_size, channel=channel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train(epoch)
        is_best = test(epoch)
        # Save sample image
        with torch.no_grad():
            sample = torch.randn(100, 40).to(device)
            sample = model.decode(sample)
            save_image(sample.view(100, -1, img_size, img_size),
                       sample_path + str(epoch) + '.png', nrow=10, normalize=True)
            if is_best:
                save_image(sample.view(100, -1, img_size, img_size),
                           sample_path + 'best_' + str(epoch) + '.png', nrow=10, normalize=True)
