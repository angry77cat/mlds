import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, word_vec=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.use_gpu = torch.cuda.is_available()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if word_vec is not None:
            self.embedding.weight.data.copy_(word_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.hidden = self.init_hidden()

    def forward(self, x, h):
        embed = self.embedding(x)
        output = embed.view(1, 1, -1)
        output, hidden = self.lstm(output, h)
        return output, hidden

    def init_hidden(self):
        # bidirectional needs "2" in the first argument
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(1, 1, self.hidden_size))
            c0 = Variable(torch.zeros(1, 1, self.hidden_size))
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
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1, word_vec=None):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        if word_vec is not None:
            self.embedding.weight.data.copy_(word_vec)
            self.embedding.weight.requires_grad = False
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        print(embedded[0].shape)
        print(hidden[0].shape)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 2)), dim=1)

        print(attn_weights[0].shape)
        print(encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


