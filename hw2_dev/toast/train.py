from preprocess import Dictionary
from model import Encoder, Decoder, AttnDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random


LR = 1e-4
EPOCH = 1
MAX_LENGTH = 10
GLOVE_DIR = 'data/word_vector/glove.6B.300d.txt'
TEACHER_RATIO = 0.5
USE_GPU = torch.cuda.is_available()



def train(x, y, dictionary, encoder, decoder, encoder_optimizer, decoder_optimizer,
          loss_func, max_length, teacher_ratio):

    # initialize hidden state
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = x.size()[0]
    output_length = y.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    decoder_input = Variable(torch.LongTensor([[dictionary("<BOS>")]]))
    if USE_GPU:
        encoder_outputs = encoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(x[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    # pass the hidden output from encoder to decoder
    decoder_hidden = encoder_hidden
    if random.random() < teacher_ratio:
        # teacher forcing!
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_func(decoder_output, y[di])
            # set the answer to the next input
            decoder_input = y[di]

    else:
        # use network's output as next input
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            if USE_GPU:
                decoder_input = decoder_input.cuda()
            loss += loss_func(decoder_output, y[di])
            # if <EOS> is generated
            if ni == dictionary("<EOS>"):
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()



if __name__ == "__main__":
    # initialize dictionary and load pretrain glove word vector
    print('loading dictionary..')
    d = Dictionary(pretrain='glove', word_vector_path=GLOVE_DIR)

    x = ['hi, how are you?', 'are you hungry?', 'wait a minute']
    y = ['I\'m fine', 'ok! let\'s eat something', 'what']

    processed_x = d.make_variable(x[0], MAX_LENGTH)
    processed_y = d.make_variable(y[0])


    # setting model
    encoder = Encoder(vocab_size=400004, hidden_size=300, word_vec=d.word_vec)
    decoder = AttnDecoder(output_size=400004, hidden_size=300, word_vec=d.word_vec, max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=LR)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=LR)

    loss_func = nn.NLLLoss()

    print('start training..')
    train(processed_x, processed_y, d, encoder, decoder, encoder_optimizer, decoder_optimizer,
          loss_func, MAX_LENGTH, 0.5)

    print('training completed!')











