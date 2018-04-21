import argparse

import torch
import torch.nn as nn

# from models import Encoder, Decoder, Seq2Seq
from preprocess import Loader


def get_args():
    parser = argparse.ArgumentParser()
    # modes
    parser.add_argument("--train", type=str, default=None, help="train mode")
    parser.add_argument("--eval", type=str, default=None, help="evaluate mode")

    # train parameters
    parser.add_argument("--epoch", type=int, default=10, help="number of epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--teacher_ratio", type=float, default=0.5, help="teacher forcing ratio")

    # evaluate parameters
    parser.add_argument("--beam", type=int, default=2, help="beam search cache size")

    return parser.parse_args()


def evaluate(model):
    model.eval()


def train(model):
    model.train()


def main():
    # get arguments
    args = get_args()

    # instantiate models
    encoder = Encoder()
    decoder = Decoder()
    seq2seq = Seq2Seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    # train
    if args.train is not None:
        train(seq2seq)

    # evaluate
    if args.eval is not None:
        eval(seq2seq)


if __name__ == "__main__":
    main()
