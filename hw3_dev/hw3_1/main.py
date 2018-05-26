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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset import ImageData
from torch.utils.data import DataLoader
from models import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np

USE_CUDA = True if torch.cuda.is_available() else False
random.seed(1234)
torch.manual_seed(1234)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('-p', '--pretrain', action='store_true', help='continue previous model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--rmsprop_lr', type= float, default= 0.002, help='learing rate, default=0.002')
    parser.add_argument('--adam_lr', type=float, default=0.00002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--ngf', type=int, default=64, help='number of layers in G')
    parser.add_argument('--ndf', type=int, default=64, help='number of layers in D')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--datadir', default= 'data/')
    parser.add_argument('--modeldir', default= 'model/')
    parser.add_argument('--outdir', default= 'output/')
    opt = parser.parse_args()
    print(opt)
    return opt

# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass
# cudnn.benchmark = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    

def train(opt):
    loss_D_hist = []
    loss_G_hist = []
    score_real_hist = []
    score_fake1_hist = []
    score_fake2_hist = []

    test = ImageData(opt)
    dataloader = DataLoader(test, batch_size= opt.batch_size, shuffle= True)
    netG = Generator(opt)
    netG.apply(weights_init)
    if opt.pretrain == True:
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)
    netD = Discriminator(opt)
    netD.apply(weights_init)
    if opt.pretrain == True:
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)
    # print(USE_CUDA)
    (netG, netD) = (netG.cuda(), netD.cuda()) if USE_CUDA else (netG, netD)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.adam_lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.adam_lr, betas=(opt.beta1, 0.999))

    fixed_noise = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    fixed_noise = fixed_noise.cuda() if USE_CUDA else fixed_noise
    real_label = 1
    fake_label = 0

    total_ite = 0
    for epoch in range(opt.epoch):
        for i, data in enumerate(dataloader, 0):
            total_ite += 1
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_input = Variable(data[0])
            real_input = real_input.cuda() if USE_CUDA else real_input
            batch_size = real_input.size(0)

            tlabel = Variable(torch.FloatTensor(torch.ones(batch_size, 1)))
            tlabel = tlabel.cuda() if USE_CUDA else tlabel
            # label = torch.full((batch_size,), real_label)
            # label = label.cuda()

            output = netD(real_input)
            errD_real = criterion(output, tlabel)
            errD_real.backward()
            D_x = output.mean()
            # print(D_x)

            # train with fake
            noise = Variable(torch.randn(batch_size, opt.nz, 1, 1))
            noise = noise.cuda() if USE_CUDA else noise
            # noise = noise.cuda()
            fake = netG(noise)
            # label.fill_(fake_label)
            flabel = Variable(torch.FloatTensor(torch.zeros(batch_size, 1)))
            flabel = flabel.cuda() if USE_CUDA else flabel
            output = netD(fake.detach())
            errD_fake = criterion(output, flabel)
            errD_fake.backward()
            D_G_z1 = output.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, tlabel)
            errG.backward()
            D_G_z2 = output.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epoch, i, len(dataloader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                vutils.save_image(real_input.data[0:25],
                        '%s/real_samples.png' % opt.outdir,
                        normalize=True, nrow= 5)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach().data[0:25],
                        '%s/fake_samples_epoch_%03d.png' % (opt.outdir, epoch),
                        normalize=True, nrow= 5)

            # recording
            loss_D_hist.append(errD.data)
            loss_G_hist.append(errG.data)
            score_real_hist.append(D_x.data)
            score_fake1_hist.append(D_G_z1.data)
            score_fake2_hist.append(D_G_z2.data)
            plt.plot(np.arange(total_ite), np.log(np.array(loss_D_hist)), label='D_loss')
            plt.plot(np.arange(total_ite), np.log(np.array(loss_G_hist)), label='G_loss')
            plt.xlabel('iteration')
            plt.ylabel('log(loss)')
            plt.legend()
            plt.savefig('%s/loss_history.png' % opt.outdir)
            plt.clf()
            # print(np.array(score_real_hist))
            plt.plot(np.arange(total_ite), np.array(score_real_hist), label='D_score')
            plt.plot(np.arange(total_ite), np.array(score_fake2_hist), label='G_score')
            plt.xlabel('iteration')
            plt.ylabel('score')
            plt.legend()
            plt.savefig('%s/score_history.png' % opt.outdir)
            plt.clf()

        torch.save(netG.state_dict(), '%s/netG_epoch_%d' % (opt.modeldir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d' % (opt.modeldir, epoch))

def eval(opt):
    fixed_noise = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    fixed_noise = fixed_noise.cuda() if USE_CUDA else fixed_noise
    netG = Generator(opt)
    netG = netG.cuda() if USE_CUDA else netG
    netG.load_state_dict(torch.load(opt.netG))
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach().data[0:25],
        '%s/fake_samples_epoch_%03d.png' % (opt.outdir, epoch),
        normalize=True, nrow= 5)


def main():
    opt = get_args()
    eval(opt) if opt.eval else train(opt)



if __name__ == '__main__':
    main()