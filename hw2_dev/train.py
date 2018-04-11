from preprocess import Dictionary, Dataset
from model import Encoder, Decoder, AttnDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


LR = 1e-4
EPOCH = 1
MAX_LENGTH = 10

d = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.300d.txt')
encoder = Encoder(vocab_size=400004, embed_size=300, hidden_size=256, batch_size=1, word_vec=d.word_vec)
decoder = Decoder(vocab_size=400004, embed_size=300, hidden_size=256, batch_size=1, output_size=400004, word_vec=d.word_vec)
if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()
x = ['hi, how are you?', 'are you hungry?', 'wait a minute']
y = ['I\'m fine', 'ok! let\'s eat something', 'what']

processed_x = [d.make_variable(i, MAX_LENGTH) for i in x]
processed_y = [d.make_variable(i, MAX_LENGTH) for i in y]


# train
encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=LR)
decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=LR)

loss_func = nn.NLLLoss()

for epoch in range(EPOCH):

    en_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    en_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size * 2))
    en_outputs = en_outputs.cuda() if torch.cuda.is_available() else en_outputs

    for i in range(MAX_LENGTH):
        en_output, en_hidden = encoder(processed_x[0][i], en_hidden)
        en_outputs[i] = en_output[0][0]

