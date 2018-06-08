import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from dataset import ImageData
from torch.utils.data import DataLoader
from models import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
import pickle

USE_CUDA = True if torch.cuda.is_available() else False
random.seed(1234)
torch.manual_seed(1234)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('-p', '--pretrain', action='store_true', help='continue previous model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('-c', '--conditional', action='store_true', default=False, help='conditional GAN')
    parser.add_argument('-w', '--wgan', action='store_true', default=False, help='WGAN')
    parser.add_argument('-wgp', '--wgan_gp', action='store_true', default=False, help='WGAN-GP')
    parser.add_argument('-n', '--noiselabel', action='store_true', default=False, help='noise label')
    parser.add_argument('--dtrain', type=int, default=1, help='number of D train')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--rmsprop_lr', type= float, default= 2e-4, help='learing rate, default=0.002')
    parser.add_argument('--adam_lr', type=float, default= 2e-5, help='learning rate, default=0.00002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ny', type=int, default=119, help='size of the latent y vector')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--ngf', type=int, default=64, help='number of layers in G')
    parser.add_argument('--ndf', type=int, default=64, help='number of layers in D')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--ec', default='', help="path to testing tags for conditional")
    parser.add_argument('--edir', default='', help="path to evaluation dir")
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
def calc_gradient_penalty(netD, real_data, fake_data, c= None):
    BATCH_SIZE = real_data.shape[0]
    # print(BATCH_SIZE)
    # print(real_data.size())

    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if USE_CUDA else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda() if USE_CUDA else interpolates
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, c)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if USE_CUDA else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # print(gradients.shape)
    gradients = gradients.view(BATCH_SIZE, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # print(gradients_norm)
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * .1
    # print(gradient_penalty)
    return 10 * ((gradients_norm - 1) ** 2).mean()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
def one_hot(x):
    # input [list]
    y = []
    for i in range(len(x)):
        y.append([0]* (x[i]) + [1] + [0] *(119 - 1- x[i]))
    return y

def train(opt):
    loss_D_hist = []
    loss_G_hist = []
    score_real_hist = []
    score_fake1_hist = []
    score_fake2_hist = []

    test = ImageData(opt, mode= 'extra')
    dataloader = DataLoader(test, batch_size= opt.batch_size, shuffle= True)
    netG = Generator(opt)
    netG.apply(weights_init)
    netD = Discriminator(opt)
    netD.apply(weights_init)
    if opt.pretrain == True:
        netG.load_state_dict(torch.load(opt.netG))
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)
    # print(USE_CUDA)
    (netG, netD) = (netG.cuda(), netD.cuda()) if USE_CUDA else (netG, netD)
    criterion = nn.BCELoss()

    if opt.wgan:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.rmsprop_lr)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.rmsprop_lr)
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.adam_lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.adam_lr, betas=(opt.beta1, 0.999))

    if opt.conditional:
        fixed_noise = Variable(torch.randn(119, opt.nz, 1, 1))
        fixed_noise = fixed_noise.cuda() if USE_CUDA else fixed_noise
        fixed_c = one_hot([i for i in range(119)])
        fixed_c = Variable(torch.FloatTensor(fixed_c)).cuda() if USE_CUDA else Variable(torch.FloatTensor(fixed_c))
    else:
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

            if opt.wgan:
            # for _ in range(opt.dtrain):
                ### train with real and fake ###
                netD.zero_grad()
                if not opt.wgan_gp:
                    for p in netD.parameters():
                        p.data.clamp_(-5e-3, 5e-3)
                real_input = Variable(data[0])
                batch_size = real_input.size(0)
                noise = Variable(torch.randn(batch_size, opt.nz, 1, 1))
                (real_input, noise) = (real_input.cuda(),noise.cuda()) if USE_CUDA else (real_input, noise)
                if opt.conditional == True:
                    c = Variable(data[1])
                    c = c.cuda() if USE_CUDA else c
                    output1 = netD(real_input, c)
                    rand_c = one_hot([random.randint(0,118) for _ in range(batch_size)])
                    rand_c = Variable(torch.FloatTensor(rand_c)).cuda() if USE_CUDA else Variable(torch.FloatTensor(rand_c))
                    fake = netG(noise, rand_c)
                    output2 = netD(fake.detach(), rand_c)
                    if opt.wgan_gp:
                        gradient_penalty = calc_gradient_penalty(netD, real_input.data, fake.data, rand_c)
                else:
                    output1 = netD(real_input)
                    fake = netG(noise)
                    output2 = netD(fake.detach())
                    if opt.wgan_gp:
                        gradient_penalty = calc_gradient_penalty(netD, real_input.data, fake.data)
                        
                if opt.wgan_gp:
                    errD = -(output1.mean() - output2.mean()) + gradient_penalty
                else:
                    errD = -(output1.mean() - output2.mean())
                # errD = -(output1.mean() - output2.mean())
                errD.backward()
                optimizerD.step()
                D_x = output1.mean()
                D_G_z1 = output2.mean()

                if (total_ite-1)%opt.dtrain  ==0:
                    netG.zero_grad()
                    output = netD(fake, rand_c) if opt.conditional else netD(fake)
                    errG = -output.mean()
                    errG.backward()
                    D_G_z2 = output.mean()
                    optimizerG.step()

            else:
                # for _ in range(opt.dtrain):
                    ### train with real ###
                netD.zero_grad()
                real_input = Variable(data[0])
                batch_size = real_input.size(0)
                noise = Variable(torch.randn(batch_size, opt.nz, 1, 1))

                if opt.noiselabel:
                    tlabel = Variable(torch.FloatTensor(torch.ones(batch_size, 1)+(torch.rand(batch_size, 1)-0.5)*0.5))
                    flabel = Variable(torch.FloatTensor(torch.zeros(batch_size, 1)+(torch.rand(batch_size, 1)*0.25)))
                else:
                    tlabel = Variable(torch.FloatTensor(torch.ones(batch_size, 1)))
                    flabel = Variable(torch.FloatTensor(torch.zeros(batch_size, 1)))

                (real_input, tlabel, noise, flabel) = (real_input.cuda(),tlabel.cuda(), noise.cuda(), flabel.cuda()) if USE_CUDA else (real_input,tlabel, noise, flabel)

                if opt.conditional == True:
                    c = Variable(data[1])
                    c = c.cuda() if USE_CUDA else c
                    output = netD(real_input, c)
                else:
                    output = netD(real_input)
                errD_real = criterion(output, tlabel)
                errD_real.backward()
                D_x = output.mean()

                ### train with fake ###
                if opt.conditional == True: 
                    rand_c = one_hot([random.randint(0,118) for _ in range(batch_size)])
                    rand_c = Variable(torch.FloatTensor(rand_c)).cuda() if USE_CUDA else Variable(torch.FloatTensor(rand_c))
                    fake = netG(noise, rand_c)
                    output = netD(fake.detach(), rand_c)
                else:
                    fake = netG(noise)
                    output = netD(fake.detach())
                errD_fake = criterion(output, flabel)
                errD_fake.backward()
                D_G_z1 = output.mean()
                errD = errD_real + errD_fake
                optimizerD.step()

                if (total_ite-1)%opt.dtrain  ==0:
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    output = netD(fake, rand_c) if opt.conditional else netD(fake)
                    errG = criterion(output, tlabel)
                    errG.backward()
                    D_G_z2 = output.mean()
                    optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.epoch, i, len(dataloader),
                    errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            if i % 50 == 0:
                vutils.save_image(real_input.data[0:25],
                        '%s/real_samples.png' % opt.outdir,
                        normalize=True, nrow= 5)
                fake = netG(fixed_noise, fixed_c) if opt.conditional else netG(fixed_noise)
                # fake = netG(fixed_noise, c)
                if opt.conditional:
                    vutils.save_image(fake.detach().data[:],
                            '%s/fake_samples_epoch_%03d.png' % (opt.outdir, epoch),
                            normalize=True, nrow= 12)
                else:
                    vutils.save_image(fake.detach().data[0:25],
                            '%s/fake_samples_epoch_%03d.png' % (opt.outdir, epoch),
                            normalize=True, nrow= 5)
                
                torch.save(netG.state_dict(), '%s/netG_epoch_%d_%i' % (opt.modeldir, epoch, i))
                torch.save(netD.state_dict(), '%s/netD_epoch_%d_%i' % (opt.modeldir, epoch, i))

            # recording
            loss_D_hist.append(errD.data)
            loss_G_hist.append(errG.data)
            score_real_hist.append(D_x.data)
            score_fake1_hist.append(D_G_z1.data)
            score_fake2_hist.append(D_G_z2.data)
            if opt.wgan:
                plt.plot(np.arange(total_ite), np.array(loss_D_hist), label='D_loss')
                plt.plot(np.arange(total_ite), np.array(loss_G_hist), label='G_loss')
            else:
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

        # torch.save(netG.state_dict(), '%s/netG_epoch_%d' % (opt.modeldir, epoch))
        # torch.save(netD.state_dict(), '%s/netD_epoch_%d' % (opt.modeldir, epoch))

def eval(opt):
    if opt.conditional:
        # load dict
        with open('tag2onehot.pkl', 'rb') as f:
            tag2onehot = pickle.load(f)
        # print(tag2onehot)

        c = []
        with open(opt.ec, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                tags = line[1].split(' ')
                c.append(tag2onehot[(tags[0], tags[2])])

        fixed_noise = Variable(torch.randn(len(c), opt.nz, 1, 1))
        fixed_noise = fixed_noise.cuda() if USE_CUDA else fixed_noise
        # fixed_c = one_hot([i for i in range(119)])
        fixed_c = one_hot(c)
        fixed_c = Variable(torch.FloatTensor(fixed_c)).cuda() if USE_CUDA else Variable(torch.FloatTensor(fixed_c))
        netG = Generator(opt)
        netG = netG.cuda() if USE_CUDA else netG
        netG.load_state_dict(torch.load(opt.netG))
        fake = netG(fixed_noise, fixed_c)
        vutils.save_image(fake.detach().data[:],
            'samples/cgan.png',
            normalize=True, nrow= 5)
    else:
        fixed_noise = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
        fixed_noise = fixed_noise.cuda() if USE_CUDA else fixed_noise
        netG = Generator(opt)
        netG = netG.cuda() if USE_CUDA else netG
        netG.load_state_dict(torch.load(opt.netG))
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach().data[0:25],
            'samples/gan.png',
            normalize=True, nrow= 5)


def main():
    opt = get_args()
    eval(opt) if opt.eval else train(opt)



if __name__ == '__main__':
    main()