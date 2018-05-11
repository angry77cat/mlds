# import torch
import numpy as np
import io

# from torch.autograd import Variable


class Vocabulary(object):

    def __init__(self):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_chars = 4
        self.max_length = 0
        self.word_list = []
        self.line_list = []

    def build_vocab(self, data_path):
        """Construct the relation between words and indices"""

        clr_convname = data_path
        with open(clr_convname) as f_txt:
            for line in f_txt:
                if line != '+++$+++':
                    line = line.strip('\n')

                    self.line_list.append(line)
                    if self.max_length < len(line):
                        self.max_length = len(line)

                    sequence = line.split()
                    for txt in sequence:
                        for ch in  txt.decode('utf-8') :
                            ch = ch.encode('utf-8')
                            if ch not in self.char2idx:
                                self.char2idx[ch] = self.num_chars
                                self.idx2char[self.num_chars] = ch
                                self.num_chars += 1
                else:
                    continue



    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2idx['SOS']] if add_sos else []
        sequence = sequence.split()
        for txt in sequence:
            for ch in  txt.decode('utf-8') :
                ch = ch.encode('utf-8')
                if ch in self.char2idx:
                    index_sequence.append( self.char2idx [ ch ] )
                else :
                    index_sequence.append( self.char2idx['UNK'] )

        if add_eos:
            index_sequence.append(self.char2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for id in indices:
            if id ==1 :
                break
            sequence += self.idx2char[id]
            
        return sequence

    def split_sequence(self, sequence):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, b, e, t]
        """
        return [char for char in sequence]

    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2char.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str


class DataTransformer(object):

    def __init__(self, path, use_cuda):
        self.indices_sequences = []
        self.target = []
        self.source = []
        self.use_cuda = use_cuda

        # Load and build the vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path)
        print 'vocab ' ,len(self.vocab.char2idx)
        self.PAD_ID = self.vocab.char2idx["PAD"]
        self.SOS_ID = self.vocab.char2idx["SOS"]
        self.vocab_size = self.vocab.num_chars
        self.max_length = self.vocab.max_length

        self._build_training_set(path)

    def _build_training_set(self, path):
        # Change sentences to indices, and append <EOS> at the end of all pairs
        # for word in self.vocab.word_list:
        #     indices_seq = self.vocab.sequence_to_indices(word, add_eos=True)
        #     # input and target are the same in auto-encoder
        #     self.indices_sequences.append([indices_seq, indices_seq[:]])
        with open (path) as f:
            get_a_batch = False
            for line in f:
                line = line.strip('\n')
                if line != '+++$+++':
                    indices_seq = self.vocab.sequence_to_indices(line, add_sos=True, add_eos=True)
                    # input and target are the same in auto-encoder

                    if get_a_batch:
                        self.indices_sequences.append([prev_indices_seq, indices_seq[1:]  ]) 
                        get_a_batch = False
                        self.source.append(prev_indices_seq)
                        self.target.append(indices_seq[1:])
                    else:
                        get_a_batch = True
                    prev_indices_seq = indices_seq

                    # if get_a_batch:
                    #     self.indices_sequences.append([prev_indices_seq, indices_seq[1:]  ]) 
                    #     get_a_batch = False
                    #     self.source.append(indices_seq)
                    #     self.target.append(indices_seq[1:])
                    # else:
                    #     get_a_batch = True
                    # prev_indices_seq = indices_seq
                else:
                    continue


    def mini_batches(self, batch_size):
        input_batches = []
        target_batches = []

        np.random.shuffle(self.indices_sequences)
        mini_batches = [
            self.indices_sequences[k: k + batch_size]
            for k in range(0, len(self.indices_sequences), batch_size)
        ]

        for batch in mini_batches:
            seq_pairs = sorted(batch, key=lambda seqs: len(seqs[0]), reverse=True) # sorted by input_lengths

            input_seqs = [pair[0] for pair in seq_pairs]
            target_seqs = [pair[1] for pair in seq_pairs]

            input_lengths = [len(s) for s in input_seqs]
            in_max = max(input_lengths)
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

            target_lengths = [len(s) for s in target_seqs]
            out_max = max(target_lengths)
            target_padded = [self.pad_sequence(s, out_max) for s in target_seqs]

            # input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch
            # target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)  # time * batch

            # if self.use_cuda:
            #     input_var = input_var.cuda()
            #     target_var = target_var.cuda()

            # input_batches.append((input_var, input_lengths))
            # target_batches.append((target_var, target_lengths))
            input_batches.append((input_padded, input_lengths))
            target_batches.append((target_padded, target_lengths))

            
        return input_batches, target_batches

    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.build_vocab('Google-10000-English.txt')
    print(vocab)

    test = "helloworld"
    print("Sequence before transformed:", test)
    ids = vocab.sequence_to_indices(test)
    print("Indices sequence:", ids)
    sent = vocab.indices_to_sequence(ids)
    print("Sequence after transformed:",sent)

    data_transformer = DataTransformer('Google-10000-English.txt', use_cuda=False)
    input_batches, target_batches = data_transformer.mini_batches(batch_size=10)
    for input_batch, target_batch in zip(input_batches, target_batches):
        print("B0-0, Inputs")
        print len( input_batch)
        print len( input_batch[2])

        print vocab.indices_to_sequence(input_batch[0][0])

        break
        # print("B0-0, Targets")
        # print(target_batch[0][0],"\n", target_batch[0][1])
