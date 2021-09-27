from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Callable
from IPython.display import HTML

from library import DS_PATH
from library.file_manager import save_torch_model

'''
nn.Sequential example :
nn.Sequential(
    nn.Linear(ndf, 512),
    nn.LeakyReLU(0.1),
    nn.Linear(512, 512),
    nn.LeakyReLU(0.1),
    nn.Linear(512, 128),
    nn.LeakyReLU(0.1),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
'''


class Generator(nn.Module):
    def __init__(self, sequential: nn.Sequential):
        super(Generator, self).__init__()
        self.main = sequential

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, sequential: nn.Sequential):
        super(Discriminator, self).__init__()
        self.main = sequential

    def forward(self, input):
        return self.main(input)


class GAN:
    dataset_gen: List
    dataset_dis: List
    labels: List
    batch_size: int
    lr: float
    beta1: float

    def __init__(
        self,
        seq_gen: nn.Sequential,
        seq_dis: nn.Sequential,
        label_size: int,
        batch_size: int = 128,
        lr: float = 0.0002,
        beta1: float = 0.5,
        criterion: Callable = nn.BCELoss,
        file_gen: str = None,
        file_dis: str = None
    ):
        print("Creating Generative Adversarial Network Object")
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.label_size = label_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(seq_gen).to(self.device)
        self.discriminator = Discriminator(seq_dis).to(self.device)

        self.criterion = criterion()
        self.nz = self.generator.main[0].in_features
        self.fixed_noise = torch.randn(self.nz, device=self.device)

        self.optimizer_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )
        self.optimizer_dis = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

        self.G_losses = []
        self.D_losses = []
        self.best_G_loss = 1000
        self.best_D_loss = 1000
        self.iters = 0

        if file_gen:
            cp_gen = torch.load(file_gen)
            self.generator.load_state_dict(cp_gen['net'])
            self.best_G_loss = cp_gen['loss']

        if file_dis:
            cp_dis = torch.load(file_dis)
            self.discriminator.load_state_dict(cp_dis['net'])
            try:
                self.best_D_loss = cp_dis['acc']
            except Exception:
                self.best_D_loss = cp_dis['loss']

    def load_pretrained(self, genpath: str, dispath: str):
        cp_gen = torch.load(genpath)
        self.generator.load_state_dict(cp_gen['net'])

        cp_dis = torch.load(dispath)
        self.discriminator.load_state_dict(cp_dis['net'])

    def prepare_data(self, dataset, labels):
        print("Preparing Dataset and Dataloader")
        self.dataset = torch.utils.data.TensorDataset(dataset, labels)
        self.dataloader = torch.utils.data.DataLoader(self.dataset)

    def train(self,
        dataset: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 300,
        savefile: str = None,
        lax_mult: float = 1.3
    ):
        self.prepare_data(dataset, labels)
        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                label = data[1][0].to(self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(self.nz, device=self.device)
                # Generate fake data batch with G
                fake = self.generator(noise)
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size, self.label_size), 0, dtype=torch.float, device=self.device)
                label = label[0]
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizer_dis.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(1) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer_gen.step()

                # Output training stats
                if (i+1) % len(self.dataloader) == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch+1, num_epochs, i+1, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    if errD.item() <= self.best_D_loss * lax_mult:
                        self.best_D_loss = min(self.best_D_loss, errD.item())

                        if savefile:
                            print('Saving Discriminator...')
                            state = {
                                'net': self.discriminator.state_dict(),
                                'loss': errD.item(),
                                'epoch': epoch,
                            }
                            save_torch_model(state, f"{DS_PATH}/data/models/GAN/{savefile}_Discriminator.pth")

                    if errG.item() <= self.best_G_loss * lax_mult:
                        self.best_G_loss = min(self.best_G_loss, errG.item())

                        if savefile:
                            print('Saving Generator...')
                            state = {
                                'net': self.generator.state_dict(),
                                'loss': errG.item(),
                                'epoch': epoch,
                            }
                            save_torch_model(state, f"{DS_PATH}/data/models/GAN/{savefile}_Generator.pth")

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()

                self.iters += 1

        return self

    def show_training_graph(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict(self, inputs: torch.Tensor) -> np.array:
        print("Prediction")
        self.discriminator.eval()
        inputs = inputs.cuda()
        predictions = []

        with torch.no_grad():
            for i, input in enumerate(inputs):
                pred = self.discriminator(input)
                pred = pred.tolist()

                predictions.append(pred)

        return np.array(predictions)
