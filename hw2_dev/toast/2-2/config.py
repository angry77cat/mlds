from gensim.models.word2vec import Word2Vec


EMBED_SIZE = 128
HIDDEN_SIZE = 256
MAX_LENGTH = 20

model = Word2Vec.load('model/word2vec.' + str(EMBED_SIZE) + 'd')
VOCAB_SIZE = len(model.wv.vocab) + 4 # include <SOS>, <EOS>, <UNK>, <PAD>
VOCAB_SIZE_prev = 198306