import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import jieba

from config import VOCAB_SIZE, MAX_LENGTH


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layer=2, dropout=0.5, bidirectional=True):
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
        word_vec = torch.FloatTensor(word_vec)
        if torch.cuda.is_available():
            word_vec = word_vec.cuda()
        self.embedding.weight.data.copy_(word_vec)
        if freeze:
            self.embedding.require_grad = False


class Decoder(nn.Module):
    """
    if attention_model is not given, the decoder will not perform attention mechanism
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layer=1, dropout=0.5, attention_model=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # load the attention model if given
        self.attention = attention_model
        if attention_model is None:
            self.gru = nn.GRU(embed_size, hidden_size, n_layer, dropout=dropout)
            self.out = nn.Linear(hidden_size, vocab_size)

        else:
            self.gru = nn.GRU(embed_size + hidden_size, hidden_size, n_layer, dropout=dropout)
            self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, source, hidden, encoder_outputs=None):
        # source [seq_len, batch_size]
        # hidden [num_layer * num_direaction, batch_size, hidden_Size]
        embedded = self.embedding(source)
        # embedded [seq_len, batch_size, embed_size]

        if self.attention is None:
            output, hidden = self.gru(embedded, hidden)
            # output [1, batch_size, hidden_size]
            output = self.out(output)
            # output [1, batch_size, vocab_size]
            output = F.log_softmax(output, dim=2)
            return output, hidden
        else:
            embedded = self.embedding(source)
            attention = self.attention(embedded, hidden, encoder_outputs)

            output = torch.cat([attention, embedded], dim=2)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            output = F.log_softmax(output, dim=2)

            return output, hidden

    def load_pretrain(self, word_vec, freeze=False):
        # load the word vector
        word_vec = torch.FloatTensor(word_vec)
        if torch.cuda.is_available():
            word_vec = word_vec.cuda()
        self.embedding.weight.data.copy_(word_vec)
        if freeze:
            self.embedding.require_grad = False


class Attention(nn.Module):
    def __init__(self, embed_size, hidden_size, max_length):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_length = max_length

        # concatenate embed and hidden, so the input size is hidden_size + embed_size
        self.linear = nn.Linear(hidden_size + embed_size, max_length)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # encoder_outputs [seq_len, batch_size, hidden_size]
        mix = torch.cat([decoder_input, decoder_hidden], dim=2)
        # decoder_input [1, batch_size, vocab_size]
        # decoder_hidden [n_layer=1, batch_size, hidden_size]
        mix = self.linear(mix)
        weight = F.softmax(mix, dim=2)
        # weight [1, batch_size, seq_len]
        weight = weight.transpose(1, 0)
        encoder_outputs = encoder_outputs.transpose(1, 0)
        # encoder_outputs [batch_size, seq_len, hidden_size]
        attention_output = torch.bmm(weight, encoder_outputs)
        # attention_output [batch_size, 1, hidden_size]
        attention_output = attention_output.transpose(1, 0)
        return attention_output


class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train(self, args, loader, dictionary):
        # set mode
        self.encoder.train()
        self.decoder.train()

        # instantiate optimizers
        encoder_optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, self.encoder.parameters()), lr=args.lr)
        decoder_optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, self.decoder.parameters()), lr=args.lr)

        for step, (x, y) in enumerate(loader):
            x, y = Variable(x), Variable(y)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # train
            loss = self.train_a_batch(x, y, dictionary, sos_idx=dictionary("<SOS>"), teacher_forcing=args.teacher_ratio)
            loss.backward()
            # clip the norm before update paremeters!
            clip_grad_norm(self.encoder.parameters(), args.grad_clip)
            clip_grad_norm(self.decoder.parameters(), args.grad_clip)

            encoder_optimizer.step()
            decoder_optimizer.step()
            # print('loss: {}'.format(loss.data[0]))

        loader.reset()
        return loss.data[0]

    def train_a_batch(self, x, y, dictionary, sos_idx=0, teacher_forcing=0.5):
        # x [seq_length, batch_size]
        # y [seq_length, batch_size]
        max_output_length = MAX_LENGTH
        batch_size = x.shape[1]
        vocab_size = self.decoder.vocab_size

        # first run encoder
        encoder_outputs, hidden = self.encoder(x)
        # pass the hidden state to decoder
        hidden = hidden[:self.decoder.n_layer]
        # feed start of sentence to decoder
        decoder_input = Variable(torch.LongTensor([[sos_idx for _ in range(batch_size)]]))
        # declare a tensor for storing decoder outputs
        decoder_outputs = Variable(torch.zeros(max_output_length, batch_size, vocab_size))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # set loss
        loss_func = nn.NLLLoss()
        for t in range(max_output_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            # decoder_output [1, batch_size, vocab_size]
            # hidden []
            decoder_outputs[t] = decoder_output[0]
            if random.random() < teacher_forcing:
                decoder_input = Variable(y.data[t].unsqueeze(0))
                # decoder_input [1, batch_size]
            else:
                decoder_input = Variable(decoder_output.data.max(2)[1])
                # decoder_input [1, batch_size]
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        # print('=========================================================')
        # print('')
        # print('')
        # print('ground truth: ', [dictionary(i.data[0]) for i in y])
        # print('rnn out: ', [dictionary(int(i)) for i in decoder_outputs.data.max(2)[1]])
        decoder_outputs = decoder_outputs.view(-1, vocab_size)

        loss = loss_func(decoder_outputs, y.view(-1))

        return loss

    # working
    def evaluate(self):
        self.encoder.eval()
        self.decoder.eval()

    def demo(self, args, dictionary):
        self.encoder.eval()
        self.decoder.eval()


        # input loop..
        while True:
            user_input = input(">>>")
            user_input = jieba.lcut(user_input)
            print(user_input)
            user_input.append("<EOS>")
            while len(user_input) < MAX_LENGTH:
                user_input.append("<PAD>")
            user_input = [dictionary.word2index.get(x, dictionary("<UNK>")) for x in user_input]
            user_input = Variable(torch.LongTensor(user_input))
            user_input = user_input.unsqueeze(1)
            if torch.cuda.is_available():
                user_input = user_input.cuda()

            # first run encoder
            encoder_outputs, hidden = self.encoder(user_input)
            # pass the hidden state to decoder
            hidden = hidden[:self.decoder.n_layer]
            # feed start of sentence to decoder
            decoder_input = Variable(torch.LongTensor([[dictionary("<SOS>")]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            answer = ""
            while True:
                decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                # decoder_output [1, batch_size, vocab_size]

                decoder_input = Variable(decoder_output.data.max(2)[1])
                next_word = dictionary(int(decoder_input.data[0]))
                if next_word == "<EOS>" or next_word == "<PAD>":
                    break
                answer += next_word
                # decoder_input [1, batch_size]
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()
            print(answer)
