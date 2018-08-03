import os
import ast
import sys
import logging
import argparse
import tensorflow as tf
import pretrainer_utils as pu
import tensorflow.contrib.eager as tfe

from dddqn import DDDQN
from encoder import EncoderRNN
from data_container import DataContainer
from pretrainer import parrot_initialization_rgc
from black_box_classifier import BlackBoxClassifier

import default  # to import os environment variables


class RGC(object):
  def __init__(self, dataset, emb_path):
    self.name = 'RGC'
    self.dataset = dataset
    self.emb_path = emb_path
    self.dc = DataContainer(dataset, emb_path)
    self.dc.prepare_data()
    self.encoder = EncoderRNN(num_units=256)
    self.dddqn = DDDQN(self.dc.word2idx, self.dc.idx2word, self.dc.idx2emb, max_tokens=self.dc.max_tokens)
    self.bbc = BlackBoxClassifier(dc=self.dc)

  def pretrain(self):
    logging.info('Launch of parrot initilization...')
    self.encoder, self.dddqn, self.dc = parrot_initialization_rgc(self.dataset, self.emb_path, dc=self.dc,
                                                                  encoder=self.encoder, dddqn=self.dddqn)

  def forward(self, x, sl, training=False):
    '''
    Performs RGC forward pass

    Inputs:
      -> x, numpy array, shape = [batch_size, input_dim], example: [batch_size, sequence_length, embedding_dim]
      -> sl, list of int, list of last sequence indice for each sample in given batch
      -> training, boolean, optional,

    Outputs:
      -> sentences, list of string, predicted sentences
    '''
    sos = self.dc.get_sos_batch_size(len(x))
    preds, logits = pu.full_encoder_dddqn_pass(x, sl, self.encoder, self.dddqn, sos, self.dc.max_tokens, training=training)
    preds = [s[:s.index(self.dc.word2idx['eos'])] for s in preds.numpy().tolist()]
    sentences = [' '.join([self.dc.idx2word[i] for i in s]) for s in preds]
    return sentences

  def test_pretained(self):
    '''
    Trains the bbc with training data, gets predictions on test data
    then gets new sentences from test data with RGC and gets predictions
    on this new test data
    print classification report for the two results
    '''
    logging.info('Train of the BBC...')
    self.bbc.train(self.bbc.x_train, self.bbc.y_train)
    logging.info('Classification report of BBC on test data...')
    self.bbc.predict_test(self.bbc.x_test, self.bbc.y_test)

    x, sl, _, _ = self.dc.transform_sources(self.bbc.x_test, emb=self.dc.emb)
    new_x_test = self.forward(x, sl)
    logging.info('Classification report of BBC on test data transformed by RGC...')
    self.bbc.predict_test(new_x_test, self.bbc.y_test)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='rgc.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
    argparser.add_argument('--key', metavar='KEY', default='test', type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    rgc = RGC(args.input, args.emb)
    rgc.pretrain()
    rgc.test_pretained()
