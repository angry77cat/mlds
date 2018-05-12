import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eval_config import MAX_LENGTH
from gensim.models import word2vec
import numpy as np


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
    def __init__(self, vocab_size, embed_size, hidden_size, attention_model= None):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        pretrained = word2vec.Word2Vec.load('stored_model/word2vec.128d')
        pretrained_vectors = np.array([pretrained[word] for word in (pretrained.wv.vocab)])
        pretrained_vectors = np.concatenate((np.random.rand(4,128)-0.5, pretrained_vectors))
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_vectors))
        self.embedding.weight.requires_grad = False

        self.attention_model= attention_model
        self.fc1 = nn.Linear(hidden_size, 1280)
        self.fc2 = nn.Linear(1280, 1280)
        self.fc3 = nn.Linear(1280, vocab_size)

        if self.attention_model is None:
            self.gru = nn.GRU(embed_size, hidden_size, num_layers=3, dropout=0.5, batch_first=True)
        else:
            self.gru = nn.GRU(embed_size + hidden_size, hidden_size, num_layers=3, dropout=0.5, batch_first=True)

    def forward(self, x, hidden, encoder_outputs=None):

        if self.attention_model is None:
            embed = self.embedding(x)
            out, hidden = self.gru(embed, hidden)
        else:   
            embed = self.embedding(x)
            attention_output = self.attention_model(embed, hidden, encoder_outputs)
            # (batch_size, 1, hidden_size) + (batch_size, 1, embed_size)
            # -> (batch_size, 1, hidden_size+ embed_size)

            out = torch.cat([attention_output, embed], dim=2)
            out = F.relu(out)
            out , hidden = self.gru(out, hidden)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=2)

        return out, hidden

class Attention(nn.Module):
    def __init__(self, embed_size, hidden_size, seq_len= 80):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.seq_len = seq_len

        # concatenate embed and hidden, so the input size is hidden_size + embed_size
        self.linear = nn.Linear(hidden_size + embed_size, seq_len)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # decoder_input: previous word(embedded), (batch_size, 1, embed_size)
        # decoder_hidden: previous hidden, (n_layer, batch_size, hidden_size)
        # encoder_outputs, (batch_size, seq_len, hidden_size)

        # (1, batch_size, embed_size)+ (1, batch_size, hidden_size) 
        # -> (1, batch_size, embed_size+ hidden_size)
        mix = torch.cat([decoder_input.transpose(0,1), 
            decoder_hidden[0,:,:].unsqueeze(0)], dim=2)

        # (1, batch_size, seq_len)
        mix = self.linear(mix)
        
        # Attention weight 
        # (1, batch_size, seq_len)
        weight = F.softmax(mix, dim=2)
        # print(weight.shape)
        # print(encoder_outputs.shape)
        # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size)
        # -> (batch_size, 1, hidden_size)
        attention_output = torch.bmm(weight.transpose(1,0), encoder_outputs)
        return attention_output