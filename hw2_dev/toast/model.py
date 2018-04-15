import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from preprocess import load_labels, load_features


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        # input is extracted features, so no embedding layer is needed

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        # set hidden state as an attribute inside the class, so no need to call it externally.
        self.hidden = self.init_hidden()

    def forward(self, x):
        x = x.view(1, 1, -1)
        out, self.hidden = self.lstm(x, self.hidden)
        return out

    def init_hidden(self):
        # bidirectional needs "2" in the first argument
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(1, 1, self.hidden_size))
            c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        return h0, c0

    def reset_hidden(self):
        self.hidden = self.init_hidden()


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.hidden = self.init_hidden()

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, 80)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, encoder_outputs):
        """
        shape of each tensor..
        encoder_outputs: 80, 50
        embedded: 1, 1, 50
        attn_weights: 1, 1, 80
        attn_applied: 1, 1, 50

        """
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((self.hidden[0], embedded), 2)), dim=2
        )
        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output[0]).unsqueeze(0)

        output = F.relu(output)
        output, self.hidden = self.lstm(output, self.hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, attn_weights

    def init_hidden(self):
        # the hidden of decoder comes from the encoder's output so this function is trivial
        return None

    def reset_hidden(self):
        self.hidden = self.init_hidden()


class Seq2Seq:
    # it is not a nn.Module!
    def __init__(self, encoder, decoder, dictionary, max_length):
        self.encoder = encoder
        self.decoder = decoder
        self.dictionary = dictionary
        self.max_length = max_length

    def train(self, encoder_optimizer, decoder_optimizer,
              loss_func, teacher_ratio, num_beam,
              train_x=None, train_y=None):
        # if user doesnt specify training data and labels
        if train_x is None or train_y is None:
            train_x, train_y = self.load_training_data()

        losses = []
        for id in range(train_x.shape[0]):
            loss = self.train_one(encoder_optimizer, decoder_optimizer,
                                  loss_func, teacher_ratio,
                                  num_beam, train_x[id], train_y[id])
            losses.append(loss)
            if id % 10 == 0:
                print('video #{:4d} | loss: {:.4f}'.format(id+1, loss))

    def train_one(self, encoder_optimizer, decoder_optimizer,
              loss_func, teacher_ratio, num_beam,
              train_x=None, train_y=None):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        self.encoder.reset_hidden()
        self.decoder.reset_hidden()

        input_length = train_x.shape[0]
        output_length = train_y.shape[0]

        encoder_outputs = Variable(torch.zeros(80, self.encoder.hidden_size))
        if torch.cuda.is_available():
            encoder_outputs = encoder_outputs.cuda()

        loss = 0
        for ei in range(input_length):
            encoder_output = self.encoder(train_x[ei])
            encoder_outputs[ei] = encoder_output[0][0]

        # pass the hidden output from encoder to decoder
        self.decoder.hidden = self.encoder.hidden
        decoder_input = Variable(torch.LongTensor([[self.dictionary("<BOS>")]]))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()

        # determine whether use ground-truth as decoder's input or not
        if random.random() < teacher_ratio:
            # teacher forcing!
            for di in range(output_length):
                decoder_output, decoder_attention = self.decoder(
                    decoder_input, encoder_outputs
                )
                loss += loss_func(decoder_output, train_y[di])
                decoder_input = train_y[di]
        else:
            # use decoder's previous output as it's input
            for di in range(output_length):
                decoder_output, decoder_attention = self.decoder(
                    decoder_input, encoder_outputs
                )
                topv, topi = decoder_output.data.topk(1)
                next = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[next]]))
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()
                loss += loss_func(decoder_output, train_y[di])
                if next == self.dictionary("<EOS>"):
                    break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / output_length

    def evaluate(self):
        raise NotImplementedError

    def load_training_data(self):
        with open("data/MLDS_hw2_1_data/training_id.txt", 'r') as f:
            train_list = [id for id in f.read().split('\n')[:-1]]
        train_x = load_features(train_list)
        train_y, max_length = load_labels(train_list, True, self.dictionary, max_length=self.max_length)
        # self.max_length = max_length
        return train_x, train_y