import argparse
import glob

import gc
import time
import torch
from gensim.models.word2vec import Word2Vec
import jieba

from models import Encoder, Decoder, Seq2Seq, Attention
from preprocess import Loader, Dictionary
from config import VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, MAX_LENGTH


def get_args():
    parser = argparse.ArgumentParser()
    # modes
    parser.add_argument("--train", action="store_true", default=False, help="train mode")
    parser.add_argument("--demo", action="store_true", default=False, help="demo mode")

    # mechanism
    parser.add_argument("-a", "--attention", action="store_true", help="use attention")

    # train parameters
    parser.add_argument("--epoch", type=int, default=10, help="number of epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--teacher_ratio", type=float, default=0.5, help="teacher forcing ratio")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="maximum of gradient norm")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")

    # evaluate parameters
    parser.add_argument("--beam", type=int, default=2, help="beam search cache size")

    return parser.parse_args()


def main():
    # get arguments
    args = get_args()

    # instantiate models
    if args.attention:
        print('use attention')
        attention = Attention(EMBED_SIZE, HIDDEN_SIZE, MAX_LENGTH)
    else:
        print('naive seq2seq')
        attention = None
    encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, dropout=args.dropout)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, dropout=args.dropout, attention_model=attention)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    seq2seq = Seq2Seq(encoder, decoder)

    # load pretrain word2vec model
    word2vec_model = Word2Vec.load('model/word2vec.%dd' % EMBED_SIZE)
    # helper class to maintain words, indexes, word vectors
    dictionary = Dictionary(word2vec_model)

    # train
    if args.train is True and args.demo is False:

        # load word vector into two models
        seq2seq.encoder.load_pretrain(dictionary.wv, freeze=False)
        seq2seq.decoder.load_pretrain(dictionary.wv, freeze=False)

        for epoch in range(args.epoch):
            args.teacher_ratio *= 1 - (epoch/args.epoch)
            args.lr -= 9e-4 * 1/args.epoch
            print('================')
            print('epoch: ', epoch)
            print('================')
            loss = 0
            start = time.time()
            for step, path in enumerate(glob.glob('data/clr/*.txt')):
                # print('training from file: ', path)
                # some training instance has just one sentence, rather than conversation..
                try:
                    loader = Loader(word2vec_model, dictionary, path, args.batch_size)
                except:
                    # print("this instance has just one sentence!")
                    continue

                # monitor variables..
                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(type(obj), obj.size())
                #     except:
                #         pass
                loss += seq2seq.train(args, loader, dictionary)
                # if step % 10 == 9:
            print('loss: {:.3f}'.format(loss/(step+1)))
            end = time.time() - start
            print("time cost: %2d:%2d:2d" % (end/3600, end/60, end%60))
                #     loss = 0

        # save models
        torch.save(attention.state_dict(), 'model/attention')
        torch.save(seq2seq.encoder.state_dict(), 'model/encoder')
        torch.save(seq2seq.decoder.state_dict(), 'model/decoder')

    # demo
    if args.demo is True and args.train is False:
        # so jieba will not show the noisy information when predicting
        _ = jieba.lcut("大家好") # dummy

        # loading pretrain model..
        print('loading pretrain model..')
        seq2seq.encoder.load_state_dict(torch.load('model/encoder'))
        seq2seq.decoder.load_state_dict(torch.load('model/decoder'))

        seq2seq.demo(args, dictionary)


if __name__ == "__main__":
    main()

