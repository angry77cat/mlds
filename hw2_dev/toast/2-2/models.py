import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from preprocess import Dictionary


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layer, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layer, dropout=dropout, bidirectional=bidirectional)

    def forward(self, source, hidden=None):
        # source [seq_len, batch_size]

        embedded = self.embedding(source)
        # embedded [seq_len, batch_size, emb_dim]

        encoder_out, encoder_hidden = self.gru(embedded, hidden)
        # encoder_out [seq_len, batch, num_direction * hidden_size]
        # encoder_hidden [num_direction * num_layer, batch_size, hidden_size]

        # bidirectional.. need to sum up
        encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        # encoder_out [seq_len, batch_size, hidden_size]

        return encoder_out, encoder_hidden

    def load_pretrain(self, word_vec, freeze=False):
        # load the word vector
        self.embedding.weight.data.copy_(word_vec)
        if freeze:
            self.embedding.require_grad = False


class Decoder(nn.Module):
    """
    if attention_model is not given, the decoder will not perform attention mechanism
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layer, dropout, attention_model=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # load the attention model if given
        self.attention = attention_model
        if attention_model is None:
            self.gru = nn.GRU(embed_size, hidden_size, n_layer, dropout=dropout)

        else:
            self.gru = nn.GRU(embed_size + hidden_size, hidden_size, n_layer, dropout=dropout)
            self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, source, hidden, encoder_outputs=None):
        # source [seq_len, batch_size]

        embedded = self.embedding(source)
        # embedded [seq_len, batch_size, embed_size]

        if self.attention is None:
            output, hidden = self.gru(embedded, hidden)
            return output, hidden
        else:
            embedded = self.embedding(source)
            attention = self.attention(embedded, hidden, encoder_outputs)

            output = torch.cat([attention, embedded], dim=1)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            output = F.softmax(output, dim=1)

            return output, hidden

    def load_pretrain(self, word_vec, freeze=False):
        # load the word vector
        self.embedding.weight.data.copy_(word_vec)
        if freeze:
            self.embedding.require_grad = False


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        # concatenate embed and hidden, so the input size is hidden_size * 2
        self.linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        mix = torch.cat([decoder_input, decoder_hidden], dim=)
        mix = self.linear(mix)
        weight = F.softmax(mix)
        attention_output = torch.bmm(weight.unsqueeze(0), encoder_outputs.unsqueeze(0))
        return attention_output


class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train(self, args, loader, dictionary):
        # set mode
        self.encoder.train()
        self.decoder.train()

        # load word vector into two models
        self.encoder.load_pretrain(dictionary.wv)
        self.decoder.load_pretrain(dictionary.wv)

        # instantiate optimizers
        encoder_optimizer = optim.Adam(params=filter(lambda x: x.require_grad(), self.encoder.parameters()), lr=args.lr)
        decoder_optimizer = optim.Adam(params=filter(lambda x: x.require_grad(), self.decoder.parameters()), lr=args.lr)


        for step, (x, y) in loader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = self.train_a_batch(x, y, sos_idx=dictionary("<SOS>"), teacher_forcing=args.teacher_ratio)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if step % 100 == 0:
                print('loss: ', loss.data[0])

        loader.reset()

    def train_a_batch(self, x, y, sos_idx=0, teacher_forcing=0.5):
        max_output_length = y.shape[0]
        batch_size = x.shape[1]
        vocab_size = self.decoder.vocab_size

        # first run encoder
        encoder_outputs, hidden = self.encoder(x)
        # pass the hidden state to decoder
        hidden = hidden[:self.decoder.n_layer]
        # feed start of sentence to decoder
        decoder_input = Variable(sos_idx)
        # declare a tensor for storing decoder outputs
        decoder_outputs = Variable(torch.zeros(max_output_length, batch_size, vocab_size))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # set loss
        loss_func = nn.NLLLoss()
        for t in range(max_output_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output
            if random.random() < teacher_forcing:
                decoder_input = Variable(y.data[t])
            else:
                decoder_input = decoder_output.data.max(1)[1]
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        loss = loss_func(decoder_outputs, y)
        return loss

    # working
    def evaluate(self):
        self.encoder.eval()
        self.decoder.eval()