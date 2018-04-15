import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, Seq2Seq
from preprocess import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=False, action='store_true', help='train mode')
args = parser.parse_args()


if __name__ == "__main__":
    # set hyperparameters
    LR = 1e-3
    TEACHER_RATIO = 0.5
    BEAM = 3
    EPOCH = 200
    MAX_LENGTH = 40
    WORD_DIM = 50

    # instantiate dictionary
    dictionary = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.50d.txt')

    # instantiate models
    encoder = Encoder(input_size=4096, hidden_size=WORD_DIM)
    decoder = Decoder(hidden_size=WORD_DIM, output_size=400004, max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # instantiate optimizers
    encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(params=decoder.parameters(), lr=LR)

    # loss function
    loss_func = nn.NLLLoss()

    # main model
    seq2seq = Seq2Seq(encoder, decoder, dictionary, MAX_LENGTH)

    # train
    seq2seq.train(encoder_optimizer=encoder_optimizer,
                  decoder_optimizer=decoder_optimizer,
                  loss_func=loss_func,
                  teacher_ratio=TEACHER_RATIO,
                  num_beam=BEAM)

    # save the model
    torch.save(seq2seq.encoder.state_dict(), 'model/encoder')
    torch.save(seq2seq.decoder.state_dict(), 'model/decoder')

    # evaluate (not implemented)
    seq2seq.evaluate()
