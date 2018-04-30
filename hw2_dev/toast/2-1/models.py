import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import MAX_LENGTH


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.fc = nn.Linear(4096, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=3, dropout=0.5, bidirectional=True, batch_first=True)

    def forward(self, x, hidden=None):
        x = self.fc(x)
        out, hidden = self.gru(x, hidden)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=3, dropout=0.5, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1280)
        self.fc2 = nn.Linear(1280, 1280)
        self.fc3 = nn.Linear(1280, vocab_size)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.gru(embed, hidden)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=2)

        return out, hidden

