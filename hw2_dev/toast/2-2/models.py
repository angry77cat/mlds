import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

    def load_pretrain(self, model, freeze=False):
        # load the word vector from 'gensim'
        self.embedding.weight.data.copy_(model.wv.syn0)
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
            # still working
            # attention_weight = self.attention(torch.cat([embedded, hidden], dim=0))
            # decoder_input = torch.bmm(attention_weight, encoder_outputs)
            # output, hidden = self.gru(torch.cat([decoder_input, embedded], dim=0), hidden)
            # output = self.out(output)
            # return output, hidden

    def load_pretrain(self, model, freeze=False):
        # load the word vector from 'gensim'
        self.embedding.weight.data.copy_(model.wv.syn0)
        if freeze:
            self.embedding.require_grad = False

# still working
class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(attention_size, attention_size, bias=False)

    def score(self, decoder_hidd):
        pass

    def forward(self, decoder_hidden, encoder_out):
        pass


class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

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

        for t in range(max_output_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output
            if random.random() < teacher_forcing:
                decoder_input = Variable(y.data[t])
            else:
                decoder_input = decoder_output.data.max(1)[1]
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        return decoder_outputs

    def train(self):
        pass