import argparse

import torch
from torch.autograd import Variable
from gensim.models.word2vec import Word2Vec

from preprocess import Dictionary, load_features
from model import Encoder, Decoder, Seq2Seq


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=40, help='')
parser.add_argument('-w', '--word_dim', type=int, default=128)
parser.add_argument('--mode', type=str, default='test')
args = parser.parse_args()


if __name__ == '__main__':
    # initialize dictionary
    WORD_DIM = args.word_dim
    model = Word2Vec.load('model/word2vec.%dd' % args.word_dim)
    vocab_size = len(model.wv.vocab) + 4
    dictionary = Dictionary(model)

    # load pretrain models
    encoder = Encoder(input_size=4096, hidden_size=WORD_DIM)
    decoder = Decoder(hidden_size=WORD_DIM, output_size=vocab_size, max_length=args.max_length)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    print("loading pretrain model..")
    encoder.load_state_dict(torch.load('model/encoder'))
    decoder.load_state_dict(torch.load('model/decoder'))

    seq2seq = Seq2Seq(encoder, decoder, dictionary)

    # load input
    with open('data/MLDS_hw2_1_data/'+args.mode+'ing_id.txt', 'r') as f:
        eval_list = [id for id in f.read().split('\n')[:-1]]
    # x = load_features(eval_list)
    # x = Variable(x[0])
    # x = Variable(torch.rand([80, 4096]))

    for id in eval_list:
        print('predicting: ', id)
        x = load_features([id], mode=args.mode)
        x = Variable(x[0])
        if torch.cuda.is_available():
            x = x.cuda()
        seq2seq.evaluate(x, id)
