import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import Encoder, Decoder, Attention
from dataset import Corpus
from config import MAX_LENGTH, VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, USE_CUDA
import matplotlib.pyplot as plt
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', default=True)
    parser.add_argument('-p', '--pretrain', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--teaching_ratio', type=float, default=0.5, help='teaching ratio')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-a', '--attention', action='store_true', default=False, help='use attention')
    return parser.parse_args()


def train(encoder, decoder, args):
    trainloss_hist= []
    testloss_hist= []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.lr)

    loss_func = nn.NLLLoss()

    #### data prepare
    corpus = Corpus(mode='test')
    loader = DataLoader(corpus, batch_size=args.batch_size, shuffle=True)
    with open('testing_data/id.txt', 'r') as f:
        test_id = [id for id in f.read().strip().split('\n')]
    test_corpus = Corpus(mode='test')
    test_loader = DataLoader(test_corpus, batch_size=100)

    #### trainging part
    for epoch in range(args.epoch):
        encoder.train()
        decoder.train()
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

            encoder_out, hidden = encoder(x)
            hidden = hidden[:3] + hidden[3:]

            decoder_input = Variable(torch.LongTensor([[1] for _ in range(x.shape[0])]))
            decoder_outputs = Variable(torch.zeros(x.shape[0], MAX_LENGTH, VOCAB_SIZE))
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
                decoder_outputs = decoder_outputs.cuda()
            losses = 0
            for t in range(MAX_LENGTH):
                output, hidden = decoder(decoder_input, hidden, encoder_out)
                decoder_outputs[:, t, :] = output
                if random.random() < args.teaching_ratio:
                    decoder_input = y[:, t].unsqueeze(1)
                else:
                    decoder_input = output.data.topk(1, dim=2)[1][:, 0]
                    decoder_input = Variable(decoder_input)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
            losses += loss_func(decoder_outputs.view(-1, VOCAB_SIZE), y.view(-1))
            losses.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        print('train loss: ', losses.data[0]/count)

        trainloss_hist.append(losses.data[0]/count)

        testloss_hist.append(evaluate(encoder, decoder, test_loader, loss_func))
        evaluate_bleu(test_id, encoder, decoder, corpus.index2word)

        score= np.load('score_history.npy')
        if score[-1:]> np.amax(score[:-1]):
            print('###score improved to '+ str(score[-1:][0])+ '###'+ '\n')
            torch.save(encoder.state_dict(), 'model/encoder')
            torch.save(decoder.state_dict(), 'model/decoder')
        else:
            print('###score not improved, highest:' +  str(np.amax(score[:-1]))+ '###')

        torch.save(encoder.state_dict(), 'model/encoder')
        torch.save(decoder.state_dict(), 'model/decoder')

        plt.plot(np.arange(epoch+1), np.log(np.array(trainloss_hist[:epoch+1])), label='trainloss')
        plt.plot(np.arange(epoch+1), np.log(np.array(testloss_hist[:epoch+1])), label='testloss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss history')
        plt.clf()
        sys.modulues


def evaluate_bleu(test_id, encoder, decoder, index2word):
    encoder.eval()
    decoder.eval()
    os.system('rm caption.txt')

    with open('caption.txt', 'a+') as f:
        for id in test_id:
            feat = np.load('testing_data/feat/'+id+'.npy')
            feat = torch.FloatTensor(feat).unsqueeze(0)
            feat = Variable(feat)
            if USE_CUDA:
                feat = feat.cuda()

            encoder_out, hidden = encoder(feat)
            hidden = hidden[:3] + hidden[3:]
            answer = ""
            decoder_input = torch.LongTensor([[1]])
            decoder_input = Variable(decoder_input)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            for _ in range(MAX_LENGTH):
                out, hidden = decoder(decoder_input, hidden, encoder_out)
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
    os.system('python self_bleu_eval.py caption.txt')


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
            output, hidden = decoder(decoder_input, hidden, encoder_out)
            decoder_outputs[:, t, :] = output

            decoder_input = output.data.topk(1, dim=2)[1][:, 0]
            decoder_input = Variable(decoder_input)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
    losses += loss_func(decoder_outputs.view(-1, VOCAB_SIZE), y.view(-1))
    print('test loss: ', losses.data[0])
    return losses.data[0]

def main():
    args = get_args()

    if args.attention:
        print('use attention')
        attention = Attention(EMBED_SIZE, HIDDEN_SIZE)
    else:
        print('naive seq2seq')
        attention = None

    # load models
    encoder = Encoder(EMBED_SIZE, HIDDEN_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE,  attention_model=attention)
    
    print(encoder)
    print(decoder)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    if args.pretrain or args.eval:
        encoder.load_state_dict(torch.load('model/encoder'))
        decoder.load_state_dict(torch.load('model/decoder'))
    else:
        np.save('score_history.npy', np.array([0]))


    # train
    if args.train:
        train(encoder, decoder, args)

if __name__ == '__main__':
    main()
