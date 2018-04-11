import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, word_vec=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_gpu = torch.cuda.is_available()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if word_vec is not None:
            self.embedding.weight.data.copy_(word_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()

    def forward(self, x, h):
        embed = self.embedding(x)
        output = embed.view(1, self.batch_size, -1)
        output, hidden = self.lstm(output, h)
        return output, hidden

    def init_hidden(self):
        # bidirectional needs "2" in the first argument
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        return h0, c0


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, output_size, word_vec=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.use_gpu = torch.cuda.is_available()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if word_vec is not None:
            self.embedding.weight.data.copy_(word_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        output = self.embedding(x).view(1, self.batch_size, -1)
        output = F.relu(output)
        output, h = self.lstm(output, h)
        output = self.softmax(self.out(output[0]))

        return output, h

    def init_hidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if self.use_gpu:
            result = result.cuda()

        return result



class AttnDecoder(nn.Module):
    def __init__(self):
        super(AttnDecoder, self).__init__()

    def forward(self, x):
        return x


class Seq2seq(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x

def train():
    pass