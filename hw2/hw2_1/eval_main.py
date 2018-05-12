import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from eval_models import Encoder, Decoder, Attention
from eval_dataset import Corpus
from eval_config import MAX_LENGTH, VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, USE_CUDA

def evaluate_bleu(test_id, encoder, decoder, index2word, feat_dir, dest_file):
    encoder.eval()
    decoder.eval()
    # os.system('rm caption.txt')

    with open(dest_file, 'a+') as f:
        for id in test_id:
            feat = np.load(feat_dir + '/feat/' + id + '.npy')
            feat = torch.FloatTensor(feat).unsqueeze(0)
            feat = Variable(feat)
            if USE_CUDA:
                feat = feat.cuda()

            encoder_out, hidden = encoder(feat)
            hidden = hidden[:3] + hidden[3:]
            answer = ""
            decoder_input = torch.LongTensor([[1]])
            decoder_input = Variable(decoder_input)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            for _ in range(MAX_LENGTH):
                out, hidden = decoder(decoder_input, hidden, encoder_out)
                topv, topi = out.data.topk(1, dim=2)
                next = topi[:, 0]
                decoder_input = Variable(next)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
                if next[0][0] == 2 or next[0][0] == 0:
                    answer = answer[:-1] + "."
                    break
                if next[0][0] == 3:
                    continue
                answer += index2word[next[0][0]] + " "

            f.write(id+','+answer+'\n')

def main():
    feat_dir = sys.argv[1]
    dest_file= sys.argv[2]

    corpus = Corpus(feat_dir)
    encoder = Encoder(EMBED_SIZE, HIDDEN_SIZE)
    attention = Attention(EMBED_SIZE, HIDDEN_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE,  attention_model=attention)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    encoder.load_state_dict(torch.load('stored_model/encoder'))
    decoder.load_state_dict(torch.load('stored_model/decoder'))
    # test(encoder, decoder, feat_dir)
    with open(feat_dir + '/id.txt', 'r') as f:
        test_id = [id for id in f.read().strip().split('\n')]
    evaluate_bleu(test_id, encoder, decoder, corpus.index2word, feat_dir, dest_file)


if __name__ == '__main__':
    main()
