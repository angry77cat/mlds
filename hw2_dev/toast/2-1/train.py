import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, Seq2Seq
from preprocess import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_length', type=int, default=40, help='max length of input')
parser.add_argument('--teaching_ratio', type=float, default=0.5, help='teaching ratio')
parser.add_argument('--word_dim', type=int, default=300, help='dimension of word embedding')

args = parser.parse_args()

if __name__ == "__main__":
    # set hyperparameters
    LR = args.lr
    TEACHER_RATIO = args.teaching_ratio
    EPOCH = args.epoch
    MAX_LENGTH = args.max_length
    WORD_DIM = args.word_dim

    # instantiate dictionary
    dictionary = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.'+str(WORD_DIM)+'d.txt', word_dim=WORD_DIM)

    # instantiate models
    encoder = Encoder(input_size=4096, hidden_size=WORD_DIM)
    decoder = Decoder(hidden_size=WORD_DIM, output_size=400004, max_length=MAX_LENGTH)
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
        seq2seq.train(encoder_optimizer=encoder_optimizer,
                      decoder_optimizer=decoder_optimizer,
                      loss_func=loss_func,
                      teacher_ratio=TEACHER_RATIO)

    # save the model
    torch.save(seq2seq.encoder.state_dict(), 'model/encoder')
    torch.save(seq2seq.decoder.state_dict(), 'model/decoder')


