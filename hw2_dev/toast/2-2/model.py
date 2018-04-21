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
        embedded = self.embedding(source)
        embedded = embedded.view(1, )
        encoder_out, encoder_hidden = self.gru(embedded, hidden)
        # bidirectional.. need to sum up
        encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        return encoder_out, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layer, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.Linear(embed_size + hidden_size, embed_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, n_layer, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, source, hidden, encoder_outputs):
        embedded = self.embedding(source)
        attention_weight = self.attention(torch.cat([embedded, hidden], dim=))
        decoder_input = torch.bmm(attention_weight, encoder_outputs)
        output, hidden = self.gru(torch.cat([decoder_input, embedded], dim=), hidden)
        output = self.out(output)
        return output, hidden

# not used
class Attention(nn.Module):
    def __init__(self, attention_size):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(attention_size, attention_size, bias=False)

    def score(self, decoder_hidd):

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing=0.5):
        batch_size = 1
        max_length = 1
        vocab_size = self.decoder.vocab_size
        decoder_outputs = Variable(torch.zeros(max_length, batch_size, vocab_size))
        if torch.cuda.is_available():
            decoder_outputs = decoder_outputs.cuda()
        encoder_outputs, hidden = self.encoder(x)
        hidden = hidden[:self.decoder.n_layer]
        decoder_input = Variable("SOS")
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
        for t in range(1, max_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output
            if random.random() < teacher_forcing:
                decoder_input = Variable(y.data[t])
            else:
                decoder_input = decoder_output.data.max(1)[1]
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        return decoder_outputs

