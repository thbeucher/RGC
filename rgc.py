import os
import ast
import sys
import logging
import argparse
import numpy as np
import utility as u
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
  def __init__(self, dataset, emb_path, name='RGC', dc=None, bbc=None, split_size=0.5):
    self.name = name
    self.dataset = dataset
    self.emb_path = emb_path
    self.get_dc(dc, split_size)
    self.encoder = EncoderRNN(num_units=256)
    self.dddqn = DDDQN(self.dc.word2idx, self.dc.idx2word, self.dc.idx2emb, max_tokens=self.dc.max_tokens)
    self.bbc = BlackBoxClassifier(dc=self.dc, prepare_data=True) if bbc is None else bbc

  def get_dc(self, dc, split_size):
    if dc is None:
      self.dc = DataContainer(self.dataset, self.emb_path, test_size=split_size)
      self.dc.prepare_data()
    else:
      self.dc = dc

  def pretrain(self):
    logging.info('Launch of parrot initilization...')
    self.encoder, self.dddqn, self.dc = parrot_initialization_rgc(self.dataset, self.emb_path, dc=self.dc,
                                                                  encoder=self.encoder, dddqn=self.dddqn)

  def update(self, rgc, init_layers=False):
    '''
    Updates rgc layers with the given RGC network
    Set init_layers to True if the RGC network you want to update is freshly instanciate
    '''
    if init_layers:
      self.encoder.init_layers()
      self.dddqn.init_layers()
    u.update_layer(self.encoder.encoder_cell, rgc.encoder.encoder_cell)
    self.dddqn.update(rgc.dddqn)

  # def get_training_format(self, x, sl, y, sos, lstm_states, preds, Qs):
  #   training = []  # list of tuple (x, sl, lstm_state, e, y, s, a, r, s', t)
  #   # lstm state = (max_step, tuple_size, batch_size, )
  #   for i, p in enumerate(preds):
  #     s = ''
  #     e = sos[i]
  #     for j, a in enumerate(p):
  #       s1 = s + self.dc.idx2word[a]
  #       lstm_state = (lstm_states[j][0][i], lstm_states[j][1][i])
  #       terminal = True if j == len(p) - 1 else False
  #       r = self.bbc.get_reward(s, y[i], terminal=terminal)
  #       experience = (x[i], sl[i], lstm_state, e, y[i], s, a, r, s1, terminal)
  #       s = s1
  #       e = self.dc.idx2emb[a]
  #       training.append(experience)
  #   return training

  def predict(self, x, sl, return_all=True):
    '''
    Performs RGC forward pass

    Inputs:
      -> x, numpy array, shape = [batch_size, input_dim], example: [batch_size, sequence_length, embedding_dim]
      -> sl, list of int, list of last sequence indice for each sample in given batch

    Outputs:
      -> sentences, list of string, predicted sentences
    '''
    sos = self.dc.get_sos_batch_size(len(x))
    preds, logits, lstm_states, Q, Qs = pu.full_encoder_dddqn_pass(x, sl, self.encoder, self.dddqn, sos, self.dc.max_tokens)
    # fill prediction with random word if only eos token in prediction, -> penalize no prediction
    preds = [s if s[0] != self.dc.word2idx['eos'] else [np.random.choice(list(self.dc.word2idx.values()))]\
             for s in preds.numpy().tolist()]
    preds = [s[:s.index(self.dc.word2idx['eos'])] if self.dc.word2idx['eos'] in s else s for s in preds]
    sentences = [' '.join([self.dc.idx2word[i] for i in s]) for s in preds]
    if return_all:
      return sentences, preds, lstm_states, Q, Qs
    else:
      return sentences

  def test_pretrained(self):
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
    new_x_test, _ = self.predict(x, sl, return_all=False)
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
  rgc.test_pretrained()
