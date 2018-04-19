import argparse

import torch
from torch.autograd import Variable

from preprocess import Dictionary, load_features
from model import Encoder, Decoder, Seq2Seq


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=40, help='')
args = parser.parse_args()


if __name__ == '__main__':
    # initialize dictionary
    WORD_DIM = 100
    dictionary = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.'+str(WORD_DIM)+'d.txt', word_dim=WORD_DIM)

    # load pretrain models
    encoder = Encoder(input_size=4096, hidden_size=WORD_DIM)
    decoder = Decoder(hidden_size=WORD_DIM, output_size=400004, max_length=args.max_length)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    print("loading pretrain model..")
    encoder.load_state_dict(torch.load('model/encoder'))
    decoder.load_state_dict(torch.load('model/decoder'))

    seq2seq = Seq2Seq(encoder, decoder, dictionary)

    # load input
    eval_list = ['_9iG5Ge01PM_3_11.avi']
    x = load_features(eval_list)
    x = Variable(x[0])
    # x = Variable(torch.rand([80, 4096]))
    if torch.cuda.is_available():
        x = x.cuda()
    seq2seq.evaluate(x)