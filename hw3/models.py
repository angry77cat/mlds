import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.conditional = args.conditional
        nz = args.nz
        ngf = args.ngf
        nc = args.nc
        ny = args.ny
        if self.conditional == False:
            self.main = nn.Sequential(
                # (nz) x 1
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # (nc) x 64 x 64
            )

        else:
            self.embed = nn.Linear(119, ny)
            self.main = nn.Sequential(
                # (nz) x 1
                nn.ConvTranspose2d(nz + ny, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # (nc) x 64 x 64
            )
    def forward(self, input, c_input= None):
        if self.conditional == False:  
            output = self.main(input)
            return output
        else:
            x = self.embed(c_input)
            x = F.relu(x)
            x = x.unsqueeze(2).unsqueeze(3)
            x1 = torch.cat([x,input],dim= 1)
            output = self.main(x1)
            return output

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.conditional = args.conditional
        self.wgan = args.wgan
        self.ndf = args.ndf
        self.ny = args.ny
        nc = args.nc
        ndf = args.ndf
        ny = args.ny
        if self.conditional == False:
            self.main = nn.Sequential(
                # (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # nn.Sigmoid()
            )
        else:
            self.embed = nn.Linear(119, ny)
            self.main = nn.Sequential(
                # (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*8) x 4 x 4
                # nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
                # nn.BatchNorm2d(ndf * 8),
                # nn.LeakyReLU(0.2, inplace=True),
                # # nn.Sigmoid()
            )
            self.c_cv = nn.ConvTranspose2d(ny, ny, 4, 1, 0, bias = False)
            self.main2 = nn.Sequential(
                nn.Conv2d(ndf * 8+ ny, ndf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.dense = nn.Linear(ndf * 8, 1)

            # self.cv = nn.Conv2d(ndf * 8 + ny, 1, 4, 1, 0, bias=False)


    def forward(self, input, c_input= None):
        if self.conditional == False:
            output = self.main(input)
            if self.wgan == False:
                output = F.sigmoid(output)
            return output.view(-1, 1).squeeze(1)
        else:
            x = self.embed(c_input)
            x = F.leaky_relu(x).unsqueeze(2).unsqueeze(3)
            x = self.c_cv(x)
            x = F.leaky_relu(x)
            output = self.main(input)
            x1 = torch.cat([x,output],dim= 1)
            x1 = self.main2(x1).view(-1, self.ndf * 8)
            x1 = self.dense(x1)
            if self.wgan == False:
                x1 = F.sigmoid(x1)
            return x1


