from gensim.models import word2vec
from collections import OrderedDict
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--make', action='store_true', default=False)
    parser.add_argument('-m', '--min_count', type=int, default=3)
    parser.add_argument('--word_dim', type=int, default=128)
    return parser.parse_args()


def make_corpus(path='data/MLDS_hw2_1_data/training_label.json'):
    f = open('data/corpus.txt', 'a+')
    data = json.load(open(path)) # a list contains dictionaries
    for dictionary in data:
        for sentence in dictionary['caption']:
            f.write(sentence+'\n')
    f.close()


def train_word_vec(args):
    sentence = word2vec.LineSentence('data/corpus.txt')
    model = word2vec.Word2Vec(sentences=sentence, size=args.word_dim, min_count=args.min_count)
    model.save('model/word2vec.%dd' % args.word_dim)
    return model


if __name__ == '__main__':
    args = get_args()

    if args.make == True:
        make_corpus('data/MLDS_hw2_1_data/training_label.json')
        make_corpus('data/MLDS_hw2_1_data/testing_label.json')
    model = train_word_vec(args)
    print('number of words: ' + str(len(model.wv.vocab)))
    with open('vocab.txt', 'w+') as f:
        vocab = {id.index+4: word for word, id in model.wv.vocab.items()}
        vocab[0] = "<PAD>"
        vocab[1] = "<SOS>"
        vocab[2] = "<EOS>"
        vocab[3] = "<UNK>"
        vocab = OrderedDict(sorted(vocab.items()))
        for id, word in vocab.items():
            f.write("{}\t{}\n".format(id, word.strip('.')))
