import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from preprocess import Loader


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(Encoder, self).__init__()

        # input is extracted features, so no embedding layer is needed

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.gru(embed, hidden)
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)
        return out, hidden

    def load_word_vec(self, dictionary):
        wv = torch.FloatTensor(dictionary.word_vec)
        if torch.cuda.is_available():
            wv = wv.cuda()
        self.embedding.weight.data.copy_(wv)


class Seq2Seq:
    # it is not a nn.Module!
    def __init__(self, encoder, decoder, dictionary=None, loader=None):
        self.encoder = encoder
        self.decoder = decoder
        self.dictionary = dictionary
        self.loader = loader

    def train(self, encoder_optimizer, decoder_optimizer,
              loss_func, teacher_ratio, batch_size):
        if self.loader is None:
            self.loader = Loader(batch_size=batch_size, dictionary=self.dictionary)
        self.loader.reset()
        losses = 0
        # for id in range(train_x.shape[0]):
        for id, (x, y) in enumerate(self.loader):
            x, y = Variable(x), Variable(y)
            loss = self.train_one(encoder_optimizer, decoder_optimizer,
                                  loss_func, teacher_ratio,
                                  x, y, batch_size)
            losses += loss
            # print('video #{:4d} | loss: {:.4f}'.format(id+1, loss))
        return losses

    def train_one(self, encoder_optimizer, decoder_optimizer,
                  loss_func, teacher_ratio,
                  train_x=None, train_y=None,
                  batch_size=None):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = train_x.shape[0]
        output_length = train_y.shape[0]

        loss = 0
        encoder_outputs, hidden = self.encoder(train_x)

        decoder_input = Variable(torch.LongTensor([[self.dictionary("<BOS>") for _ in range(batch_size) ]]))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()

        # determine whether use ground-truth as decoder's input or not
        if random.random() < teacher_ratio:
            # teacher forcing!
            for di in range(output_length):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                loss += loss_func(decoder_output, train_y[di])
                decoder_input = train_y[di]
        else:
            # use decoder's previous output as it's input
            for di in range(output_length):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                print(decoder_output.shape)
                topv, topi = decoder_output.data.topk(1, dim=2)
                next = topi[:, :, 0]
                decoder_input = Variable(next)
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()
                print(decoder_output, train_y[di])
                loss += loss_func(decoder_output, train_y[di])


        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / output_length

    def evaluate(self, x, id):
        self.encoder.reset_hidden()
        self.decoder.reset_hidden()

        input_length = x.shape[0]

        encoder_outputs = Variable(torch.zeros(80, self.encoder.hidden_size))
        if torch.cuda.is_available():
            encoder_outputs = encoder_outputs.cuda()

        loss = 0
        for ei in range(input_length):
            encoder_output = self.encoder(x[ei])
            encoder_outputs[ei] = encoder_output[0][0]

        # pass the hidden output from encoder to decoder
        self.decoder.hidden = self.encoder.hidden
        decoder_input = Variable(torch.LongTensor([[self.dictionary("<BOS>")]]))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()

        answer = ""
        while True:
            decoder_output, decoder_attention = self.decoder(
                decoder_input, encoder_outputs
            )
            topv, topi = decoder_output.data.topk(1)
            next = topi[0][0]
            # print(self.dictionary(next))
            decoder_input = Variable(torch.LongTensor([[next]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            # loss += loss_func(decoder_output, train_y[di])
            if next == self.dictionary("<EOS>") or next == self.dictionary("<PAD>"):
                answer = answer[:-1] + "."
                break
            if next == self.dictionary("<UNK>"):
                continue
            answer += self.dictionary(next)
            answer += " "
        with open("data/MLDS_hw2_1_data/caption.txt", 'a+') as f:
            f.write(id + ',' + answer + '\n')

    # deprecated
    # def load_training_data(self):
    #     with open("data/MLDS_hw2_1_data/training_id.txt", 'r') as f:
    #         train_list = [id for id in f.read().split('\n')[:-1]]
    #     train_x = load_features(train_list[:200])
    #     train_y, max_length = load_labels(train_list[:200], True, self.dictionary, max_length=self.max_length)
    #     # self.max_length = max_length
    #     return train_x, train_y
