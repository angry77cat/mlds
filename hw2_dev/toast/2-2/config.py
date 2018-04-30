from gensim.models.word2vec import Word2Vec


EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_LENGTH = 15

# model = Word2Vec.load('model/word2vec.' + str(EMBED_SIZE) + 'd')
VOCAB_SIZE = 3000
# VOCAB_SIZE = len(model.wv.vocab) + 4 # include <SOS>, <EOS>, <UNK>, <PAD>
# VOCAB_SIZE = 15855 # include <SOS>, <EOS>, <UNK>, <PAD>
VOCAB_SIZE_prev = 198306
