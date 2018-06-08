import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from acgan_model import D, G
import random
from acgan_dataset import ImageData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



USE_CUDA = True if torch.cuda.is_available() else False
random.seed(1234)
torch.manual_seed(1234)

def get_args():
    parser = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')

    parser.add_argument('--datadir', default= 'data/')
    parser.add_argument('--modeldir', default= 'model/')
    parser.add_argument('--outdir', default= 'output/')
    parser.add_argument('--manual_seed', default=42, help='manual seed.')
    parser.add_argument('--image_size', default=64, help='image size.')
    parser.add_argument('--batch_size', default=64, help='batch size.')
    parser.add_argument('--nz', default=100, help='length of noize.')
    parser.add_argument('--ndf', default=64, help='number of filters.')
    parser.add_argument('--ngf', default=64, help='number of filters.')
    opt = parser.parse_args()
    print(opt)
    return opt



def train(opt):
    loss_D_hist = []
    loss_G_hist = []
    score_real_hist = []
    score_fake1_hist = []
    score_fake2_hist = []

    test = ImageData(opt, mode= 'extra')
    dataloader = DataLoader(test, batch_size= opt.batch_size, shuffle= True)

    bce = nn.BCELoss()
    cep = nn.CrossEntropyLoss()
    if USE_CUDA:
        bce = bce.cuda()
        cep = cep.cuda()

    netG = G(opt)
    netD = D(opt)
    if USE_CUDA:
        netD = netD.cuda()
        netG = netG.cuda()

    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    embed = nn.Embedding(119, opt.nz)
    label = Variable(torch.LongTensor([range(119)]*1)).view(-1)
    fixed = Variable(torch.Tensor(119, opt.nz).normal_(0, 1))
    if USE_CUDA:
        embed = embed.cuda()
        label = label.cuda()
        fixed = fixed.cuda()
    fixed.mul_(embed(label))

    total_ite = 0
    netG.train()
    netD.train()
    for epoch in range(opt.epoch):
        for i, data in enumerate(dataloader, 0):
            total_ite += 1

            # true

            real_input = Variable(data[0])
            real_c = Variable(data[1])
            real_ = Variable(torch.ones(real_c.size()))
            if USE_CUDA:
                real_input = real_input.cuda()
                real_c = real_c.cuda()
                real_ = real_.cuda()
            # fake
            noise = Variable(torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1))
            fake_c = Variable(torch.LongTensor(opt.batch_size).random_(119))
            if USE_CUDA:
                noise = noise.cuda()
                fake_c = fake_c.cuda()
            noise.mul_(embed(fake_c))
            # noise = noise.unsqueeze(2).unsqueeze(3)
            # print(noise.shape)
            fake_ = Variable(torch.zeros(fake_c.size()))
            real_ = Variable(torch.ones(fake_c.size()))
            if USE_CUDA:
                fake_ = fake_.cuda()
                real_ = real_.cuda()

            # update D
            netD.zero_grad()
            fake_input = netG(noise)
            real_pred, real_cls = netD(real_input)
            fake_pred, fake_cls = netD(fake_input.detach())
            # CrossEntropyLoss: Input(N,C)/Target(C)
            real_loss = bce(real_pred, real_) + cep(real_cls, real_c.squeeze(1)) * 10
            fake_loss = bce(fake_pred, fake_) + cep(fake_cls, fake_c) * 10
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optD.step()

            # update G
            netG.zero_grad()
            fake_pred2, fake_cls = netD(fake_input)
            g_loss = bce(fake_pred2, real_) + cep(fake_cls, fake_c) * 10
            g_loss.backward()
            optG.step()


            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.epoch, i, len(dataloader),
                    d_loss.data, g_loss.data, real_pred.mean(), fake_pred.mean(), fake_pred2.mean()))

            if i % 50 == 0:
                netG.eval()
                fixed_input = netG(fixed)
                vutils.save_image(fixed_input.detach().data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outdir, epoch),
                    normalize= True,nrow=12)
                torch.save(netG.state_dict(), '%s/netG_epoch_%d_%i' % (opt.modeldir, epoch, i))
                torch.save(netD.state_dict(), '%s/netD_epoch_%d_%i' % (opt.modeldir, epoch, i))

            loss_D_hist.append(d_loss.data)
            loss_G_hist.append(g_loss.data)
            score_real_hist.append(real_pred.mean().data)
            score_fake1_hist.append(fake_pred.mean().data)
            score_fake2_hist.append(fake_pred2.mean().data)
            # if opt.wgan:
            #     plt.plot(np.arange(total_ite), np.array(loss_D_hist), label='D_loss')
            #     plt.plot(np.arange(total_ite), np.array(loss_G_hist), label='G_loss')
            # else:
            plt.plot(np.arange(total_ite), np.log(np.array(loss_D_hist)), label='D_loss')
            plt.plot(np.arange(total_ite), np.log(np.array(loss_G_hist)), label='G_loss')
            plt.xlabel('iteration')
            plt.ylabel('log(loss)')
            plt.legend()
            plt.savefig('%s/loss_history.png' % opt.outdir)
            plt.clf()

def one_hot(x):
    y = []
    for i in range(len(x)):
        y.append([0]* (x[i]) + [1] + [0] *(119 - 1- x[i]))
    return y
# tsfm=transforms.Compose([
#     transforms.Resize(opt.image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# if opt.dataset == 'cifar10':
#     dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)
# elif opt.dataset == 'mnist':
#     dataset = dset.MNIST(root=opt.dataroot, download=True, train=True, transform=tsfm)

# loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

# bce = nn.BCELoss().cuda()
# cep = nn.CrossEntropyLoss().cuda()

# opt.nc = 1 if opt.dataset == 'mnist' else 3

# netd = D(ndf=opt.ndf, nc=opt.nc, num_classes=10).cuda()
# netg = G(ngf=opt.ngf, nc=opt.nc, nz=opt.nz).cuda()

# optd = optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optg = optim.Adam(netg.parameters(), lr=0.0002, betas=(0.5, 0.999))

# embed = nn.Embedding(10, opt.nz).cuda()
# label = Variable(torch.LongTensor([range(10)]*10)).view(-1).cuda()
# fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
# fixed.mul_(embed(label))



# def denorm(x):
    # return x * 0.5 + 0.5

# def train2(epoch):
#     netg.train()
#     netd.train()
#     for _, (image, label) in enumerate(loader):
#         #######################
#         # real input and label
#         #######################
#         # real_input = Variable(image).cuda()
        # real_label = Variable(label).cuda()
        # real_ = Variable(torch.ones(real_label.size())).cuda()

        # #######################
        # # fake input and label
        # #######################
        # noise = Variable(torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1)).cuda()
        # fake_label = Variable(torch.LongTensor(opt.batch_size).random_(10)).cuda()
        # noise.mul_(embed(fake_label))
        # fake_ = Variable(torch.zeros(fake_label.size())).cuda()

        # #######################
        # # update net d
        # #######################
        # netd.zero_grad()
        # fake_input = netg(noise)

        # real_pred, real_cls = netd(real_input)
        # fake_pred, fake_cls = netd(fake_input.detach())

        # real_loss = bce(real_pred, real_) + cep(real_cls, real_label) * 10
        # fake_loss = bce(fake_pred, fake_) + cep(fake_cls, fake_label) * 10
        # d_loss = real_loss + fake_loss
        # d_loss.backward()
        # optd.step()

        #######################
        # update net g
        #######################
        # optg.zero_grad()
        # fake_pred, fake_cls = netd(fake_input)
        # real_ = Variable(torch.ones(fake_label.size())).cuda()
        # g_loss = bce(fake_pred, real_) + cep(fake_cls, fake_label) * 10
        # g_loss.backward()
        # optg.step()

#     #######################
#     # save image pre epoch
#     #######################
#     utils.save_image(denorm(fake_input.data), f'images/fake_{epoch:03d}.jpg')
#     utils.save_image(denorm(real_input.data), f'images/real_{epoch:03d}.jpg')

#     #######################
#     # save model pre epoch
#     #######################
#     torch.save(netg, f'chkpts/g_{epoch:03d}.pth')
#     torch.save(netd, f'chkpts/d_{epoch:03d}.pth')


# def test(epoch):
#     netg.eval()

#     fixed_input = netg(fixed)

#     utils.save_image(denorm(fixed_input.data), f'images/fixed_{epoch:03d}.jpg', nrow=10)

def main():
    opt = get_args()
    eval(opt) if opt.eval else train(opt)

if __name__ == '__main__':
    main()
#     # for epoch in range(opt.num_epoches):
#     #     print(f'Epoch {epoch:03d}.')
#     #     train(epoch)
#     #     test(epoch)