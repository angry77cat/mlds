import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import Encoder, Decoder
from dataset import Corpus
from config import MAX_LENGTH, VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, USE_CUDA


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', default=True)
    parser.add_argument('-p', '--pretrain', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--teaching_ratio', type=float, default=0.5, help='teaching ratio')
    parser.add_argument('-b', '--batch_size', type=int, default=32)

    return parser.parse_args()


def train(encoder, decoder, args):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    corpus = Corpus(mode='train')
    loader = DataLoader(corpus, batch_size=args.batch_size, shuffle=True)
    loss_func = nn.NLLLoss()

    # for validset
    with open('testing_id.txt', 'r') as f:
        test_id = [id for id in f.read().strip().split('\n')]
    test_corpus = Corpus(mode='test')
    test_loader = DataLoader(test_corpus, batch_size=100)

    for epoch in range(args.epoch):
        encoder.train()
        decoder.train()
        start = time.time()
        print('epoch: ', epoch)
        count = 0
        for x, y in loader:
            count += 1
            x, y = Variable(x), Variable(y)
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #############
            # main part #
            #############
            encoder_out, hidden = encoder(x)
            hidden = hidden[:3] + hidden[3:]

            decoder_input = Variable(torch.LongTensor([[1] for _ in range(x.shape[0])]))
            decoder_outputs = Variable(torch.zeros(x.shape[0], MAX_LENGTH, VOCAB_SIZE))
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
                decoder_outputs = decoder_outputs.cuda()
            losses = 0
            for t in range(MAX_LENGTH):
                output, hidden = decoder(decoder_input, hidden)
                decoder_outputs[:, t, :] = output
                if random.random() < args.teaching_ratio:
                    decoder_input = output.data.topk(1, dim=2)[1][:, 0]
                    decoder_input = Variable(decoder_input)
                else:
                    decoder_input = y[:, t].unsqueeze(1)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
            losses += loss_func(decoder_outputs.view(-1, VOCAB_SIZE), y.view(-1))
            #############
            # main part #
            #############
            losses.backward()
            # if count % 10 == 0:
            #     print("step: {} | loss: {}".format(count, losses.data[0]))
            encoder_optimizer.step()
            decoder_optimizer.step()
        print('train loss: ', losses.data[0]/count)
        end = time.time() - start
        # print('time: %2d:%2d' % (end//60, end % 60))
        evaluate(encoder, decoder, test_loader, loss_func)
        evaluate_bleu(test_id, encoder, decoder, corpus.index2word)
        torch.save(encoder.state_dict(), 'model/encoder')
        torch.save(decoder.state_dict(), 'model/decoder')


def evaluate_bleu(test_id, encoder, decoder, index2word):
    encoder.eval()
    decoder.eval()
    os.system('rm caption.txt')

    with open('caption.txt', 'a+') as f:
        for id in test_id:
            feat = np.load('data/MLDS_hw2_1_data/testing_data/feat/'+id+'.npy')
            feat = torch.FloatTensor(feat).unsqueeze(0)
            feat = Variable(feat)
            if USE_CUDA:
                feat = feat.cuda()

            out, hidden = encoder(feat)
            hidden = hidden[:3] + hidden[3:]
            answer = ""
            decoder_input = torch.LongTensor([[1]])
            decoder_input = Variable(decoder_input)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            while True:
                out, hidden = decoder(decoder_input, hidden)
                topv, topi = out.data.topk(1, dim=2)
                next = topi[:, 0]
                decoder_input = Variable(next)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
                if next[0][0] == 2 or next[0][0] == 0:
                    answer = answer[:-1] + "."
                    break
                if next[0][0] == 3:
                    continue
                answer += index2word[next[0][0]] + " "

            f.write(id+','+answer+'\n')
    os.system('python bleu_eval.py caption.txt')


def evaluate(encoder, decoder, loader, loss_func):
    encoder.eval()
    decoder.eval()
    for x, y in loader:
        x, y = Variable(x), Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()
        encoder_out, hidden = encoder(x)
        hidden = hidden[:3] + hidden[3:]

        decoder_input = Variable(torch.LongTensor([[1] for _ in range(x.shape[0])]))
        decoder_outputs = Variable(torch.zeros(x.shape[0], MAX_LENGTH, VOCAB_SIZE))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()
        losses = 0
        for t in range(MAX_LENGTH):
            output, hidden = decoder(decoder_input, hidden)
            decoder_outputs[:, t, :] = output

            decoder_input = output.data.topk(1, dim=2)[1][:, 0]
            decoder_input = Variable(decoder_input)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
    losses += loss_func(decoder_outputs.view(-1, VOCAB_SIZE), y.view(-1))
    print('test loss: ', losses.data[0])

def main():
    args = get_args()

    # load models
    encoder = Encoder(EMBED_SIZE, HIDDEN_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    if args.pretrain or args.eval:
        encoder.load_state_dict(torch.load('model/encoder'))
        decoder.load_state_dict(torch.load('model/decoder'))

    # train
    if args.train:
        train(encoder, decoder, args)

    # evaluate
    # if args.eval:
    #     eval()


if __name__ == '__main__':
    main()
