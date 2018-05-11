import os
import re
import sys
import logging
import time
import random
import math
import pickle as pk

import nltk
import tensorflow as tf
import numpy as np
from numpy.random import multinomial

import data_readerv2
import DataHelper_seq as datahelper
import at_seq2seq_model

input_file  = sys.argv[1]
output_file = sys.argv[2]

tf.app.flags.DEFINE_boolean("train", False, "True for training False to testing")
FLAGS = tf.app.flags.FLAGS

_bucket = [(5, 10), (10, 15), (20, 25), (40, 50)]
def create_model(sess, forward_only, word_dict, inv_word_dict):
    model = at_seq2seq_model.chatbot(
        word_dict=word_dict,
        inv_word_dict=inv_word_dict,
        source_vocab_size=len(word_dict),
        target_vocab_size=len(word_dict),
        buckets=_bucket,
        size=512,
        num_layers=2,
        max_gradient_norm=5.0,
        batch_size=64,
        learning_rate=0.001,
        learning_rate_decay_factor=0.99,
        use_lstm=False,
        num_samples=512,
        forward_only=forward_only,
        dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    return model

def train():
    with tf.Session() as sess:
        #  w_id, inv_w_id, train_set = data_readerv2.read_chatter()
        print 'load word dict......'

        # vocab = datahelper.Vocabulary()
        # vocab.build_vocab('clr_conversation.txt')   
        # w_id = vocab.char2idx
        # inv_w_id = vocab.idx2char
        # pk.dump(w_id,open('w_id.pk','w'))
        # pk.dump(inv_w_id, open('inv_w_id.pk','w'))

        w_id = pk.load(open('w_id.pk','r'))
        inv_w_id = pk.load(open('inv_w_id.pk', 'r'))

        traindata = datahelper.DataTransformer('clr_conversation.txt', use_cuda=True)
        
        print 'finish load word dict......'
        #2785954
        train_set = data_readerv2.read_lines(w_id ,'clr_conversation.txt',traindata, 2700000)
        #print len(train_set)
        #print len(train_set[0])
        model = create_model(sess, False, w_id, inv_w_id)

        model.load(sess, './at_s2s_model/')
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_bucket))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print "-----start training-----"
        debug = False
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                w_id ,train_set, bucket_id, model.batch_size)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / 100
            loss += step_loss / 100
            current_step += 1
            if current_step % 100 == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f loss "
                       "%f" % (model.global_step.eval(), model.learning_rate.eval(),
                        step_time, loss))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = './s2s_model/'
                model_name = './model'
                model.saver.save(
                    sess, os.path.join(checkpoint_path, model_name), 
                    model.global_step)
                step_time, loss = 0.0, 0.0
def test():
    with tf.Session() as sess:
        print 'run test......'
        print 'load word dict......'

        # vocab = datahelper.Vocabulary()
        # vocab.build_vocab('clr_conversation.txt')   
        # w_id = vocab.char2idx
        # inv_w_id = vocab.idx2char
        # pk.dump(w_id,open('w_id.pk','w'))
        # pk.dump(inv_w_id, open('inv_w_id.pk','w'))

        w_id = pk.load(open('w_id.pk','r'))
        inv_w_id = pk.load(open('inv_w_id.pk', 'r'))

        print 'finish load word dict......'
        #  train_set = data_readerv2.read_lines(w_id, './data/chat.txt', 0)
        model = create_model(sess, True, w_id, inv_w_id)
        model.load(sess, './s2s_model/')
        model.batch_size = 1
        print '-----start testing-----'
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        
                
        def s2id(s):
            s_id = []
            for w in s:
                if w in w_id:
                    s_id.append(w_id[w])
                else:
                    s_id.append(0)
            return s_id

        def id2s(ids):
            s = ""
            for w in ids:
                if w != 0:
                    s += inv_w_id[w]
                    s += " "
                if w == w_id['EOS']:
                    break
            return s

        while sentence:
            sequence = sentence.split()
            #print sequence
            ch_sentence= [0]
            for txt in sequence:
                for ch in  txt.decode('utf-8') :
                    ch = ch.encode('utf-8')
                    ch_sentence.append(ch)
            #print ch_sentence

            token_ids = s2id(ch_sentence)
            #print token_ids
            bucket_id = len(_bucket) - 1
            for i, bucket in enumerate(_bucket):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", ch_sentence)
            encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                w_id, {bucket_id: [(token_ids, [])]}, bucket_id, model.batch_size)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                              target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            #  outputs = [int(multinomial(20, logit, 1)) for logit in output_logits]
            print id2s(outputs)
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def baseline():
    with tf.Session() as sess:
        print 'run baseline output.....'
        print 'load word dict......'

        w_id = pk.load(open('w_id.pk','r'))
        inv_w_id = pk.load(open('inv_w_id.pk', 'r'))

        print 'finish load word dict......'
        #  train_set = data_readerv2.read_lines(w_id, './data/chat.txt', 0)
        model = create_model(sess, True, w_id, inv_w_id)
        model.load(sess, './s2s_model/')
        model.batch_size = 1
        print '-----start testing-----'

        def s2id(s):
            s_id = []
            for w in s:
                if w in w_id:
                    s_id.append(w_id[w])
                else:
                    s_id.append(0)
            return s_id

        def id2s(ids):
            s = ""
            for w in ids:
                if w != 0:
                    s += inv_w_id[w]
                    s += " "
                if w == w_id['EOS']:
                    break
            return s

        #data_path= 'baseline/evaluation/'
        with open( input_file,'r') as base_input:
            with open( output_file ,'w') as base_output:
                for line in base_input:
                    line = line.strip('\n')

                    sequence = line.split()
                    ch_sentence= [0]
                    for txt in sequence:
                        for ch in  txt.decode('utf-8') :
                            ch = ch.encode('utf-8')
                            ch_sentence.append(ch)

                    token_ids = s2id(ch_sentence)
                    bucket_id = len(_bucket) - 1
                    for i, bucket in enumerate(_bucket):
                        if bucket[0] >= len(token_ids):
                            bucket_id = i
                            break
                    else:
                        logging.warning("Sentence truncated: %s", ch_sentence)
                    encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                        w_id, {bucket_id: [(token_ids, [])]}, bucket_id, model.batch_size)
                    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                    target_weights, bucket_id, True)
                    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                    #  outputs = [int(multinomial(20, logit, 1)) for logit in output_logits]
                    print id2s(outputs)
                    base_output.write( id2s(outputs)[:-1]+'\n' )


        
def main(_):
    if FLAGS.train:
        train()
    else:
        baseline()
        #test()

if __name__ == "__main__":
    tf.app.run()
