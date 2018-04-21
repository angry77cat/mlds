import argparse
import logging
from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_dim", type=int, default=100, help="dimension of word embedding")

    return parser.parse_args()

def make_pair(x, y):
    pass




if __name__ == "__main__":
    # get arguments
    args = get_args()

    # load data
    with open("data/clr_conversation.txt", 'r') as f:
        pass

    sentences = []
    logging.info("training word embedding..")
    # model = word2vec.Word2Vec(sentences, size=args.word_dim)
    logging.info("completed training word embedding!")
