import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

from model import Encoder, Decoder, Seq2Seq
from preprocess import Dictionary


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_length', type=int, default=10, help='max length of input')
parser.add_argument('--teaching_ratio', type=float, default=0.5, help='teaching ratio')
parser.add_argument('-w', '--word_dim', type=int, default=256, help='dimension of word embedding')
parser.add_argument('-p', '--pretrain', action='store_true')
parser.add_argument('-b', '--batch_size', type=int, default=32)

args = parser.parse_args()

if __name__ == "__main__":
    # set hyperparameters
    LR = args.lr
    TEACHER_RATIO = args.teaching_ratio
    EPOCH = args.epoch
    MAX_LENGTH = args.max_length
    WORD_DIM = args.word_dim

    # instantiate dictionary
    model = Word2Vec.load('model/word2vec.%dd' % args.word_dim)
    # wv = KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin.gz', binary=True)
    vocab_size = len(model.wv.vocab) + 4
    dictionary = Dictionary(model)

    # instantiate models
    encoder = Encoder(input_size=4096, hidden_size=WORD_DIM)
    decoder = Decoder(hidden_size=WORD_DIM, vocab_size=vocab_size, embed_size=args.word_dim)
    if args.pretrain:
        encoder.load_state_dict(torch.load('model/encoder'))
        decoder.load_state_dict(torch.load('model/decoder'))
    else:	
        decoder.load_word_vec(dictionary)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # instantiate optimizers
    encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(params=decoder.parameters(), lr=LR)

    # loss function
    loss_func = nn.NLLLoss()

    # main model
    seq2seq = Seq2Seq(encoder, decoder, dictionary)

    # train
    for i in range(EPOCH):
        print('epoch: ', i)
        start = time.time()
        loss = seq2seq.train(encoder_optimizer=encoder_optimizer,
                             decoder_optimizer=decoder_optimizer,
                             loss_func=loss_func,
                             teacher_ratio=TEACHER_RATIO,
                             batch_size=args.batch_size)
        end = time.time() - start
        print("loss: ", loss/1450*args.batch_size)
        # print("time: %2d:%2d" % (end/60, end%60))
        # save the model
        torch.save(seq2seq.encoder.state_dict(), 'model/encoder')
        torch.save(seq2seq.decoder.state_dict(), 'model/decoder')


