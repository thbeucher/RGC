import os
import ast
import sys
import logging
import argparse
import numpy as np
import utility as u
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from rgc import RGC


import default


def eval_rgc(epoch, dqn, x_test, sl_test, labels, dls_test, report=False, show_idx=None):
  rgc_sentences = dqn.predict(x_test, sl_test, return_all=False)
  if show_idx is not None:
    with open('rgc_results_extract.txt', 'a') as f:
      f.write('Epoch {}:\n'.format(epoch))
      dtnp = np.asarray(dls_test)[show_idx]
      snp = np.asarray(rgc_sentences)[show_idx]
      lnp = np.asarray(labels)[show_idx]
      for t, p, l in zip(dtnp, snp, lnp):
        f.write('Label -> ' + l + '\nTarget -> ' + t + '\nPred -> ' + p + '\n\n')
      f.write('\n\n\n\n')
  if report:
    dqn.bbc.predict_test(rgc_sentences, labels)
  else:
    acc, f1 = dqn.bbc.get_accuracy_f1(rgc_sentences, labels)
    return acc, f1


def main_execution(dataset, emb_path):
  DQNetwork = RGC(dataset, emb_path, name='DQNetwork')

  dls = [el.replace(' eos', '') for el in DQNetwork.dc.dls]  # remove eos token
  (_, x_rgc, x_te), (_, sl_rgc, sl_te), (dls_bbc, dls_rgc, dls_te), (y_bbc, y_rgc, y_te)\
    = u.split_into_3(DQNetwork.dc.x, DQNetwork.dc.sl, dls, DQNetwork.dc.labels)
  x_rgc_batch, sl_rgc_batch, y_rgc_batch = u.to_batch(x_rgc, sl_rgc, y_rgc, batch_size=DQNetwork.dc.batch_size)
  # train bbc
  DQNetwork.bbc.train(dls_bbc, y_bbc)
  # eval bbc
  mispredicted, f1_ref = DQNetwork.bbc.predict_test(dls_te, y_te)

  TargetNetwork = RGC(dataset, emb_path, dc=DQNetwork.dc, bbc=DQNetwork.bbc, name='TargetNetwork')

  # initialize networks
  DQNetwork.pretrain()
  TargetNetwork.update(DQNetwork, init_layers=True)
  # eval rgc pretrained
  eval_rgc('None', DQNetwork, x_te, sl_te, y_te, dls_te, report=True)

  def get_loss(dqn, x, sl, y, Qs_t, gamma, sos):
    sentences, preds, lstm_states, Q, Qs = dqn.predict(x, sl)
    training = []
    batch_loss = []
    for i, p in enumerate(preds):
      s = ''
      e = sos[i]
      sentence_losses = []
      for j, a in enumerate(p):
        s1 = s + dqn.dc.idx2word[a]
        if j == len(p) - 1:
          r = dqn.bbc.get_reward(s, y[i], terminal=True)
          q_next = r
        else:
          r = dqn.bbc.get_reward(s, y[i])
          q_next = r + gamma * Qs_t[i][j][a]
        loss = tf.square(q_next - Q[i])
        sentence_losses.append(loss)

        s = s1
        e = dqn.dc.idx2emb[a]
      batch_loss.append(tf.reduce_sum(sentence_losses))
    final_loss = tf.reduce_mean(batch_loss)
    logging.info('loss = {}'.format(final_loss))
    return final_loss

  optimizer = tf.train.AdamOptimizer()
  gamma = 0.7
  sos = DQNetwork.dc.get_sos_batch_size(len(x_rgc))
  for epoch in range(300):
    for x, sl, y in zip(x_rgc_batch, sl_rgc_batch, y_rgc_batch):
      _, _, _, _, Qs_t = TargetNetwork.predict(x, sl)
      optimizer.minimize(lambda: get_loss(DQNetwork, x, sl, y, Qs_t, gamma, sos))
    _, training_f1 = eval_rgc(epoch, DQNetwork, x_rgc, sl_rgc, y_rgc, dls_rgc)
    logging.info('Training f1 on epoch {} = {}'.format(epoch, training_f1))
    _, validation_f1 = eval_rgc(epoch, DQNetwork, x_te, sl_te, y_te, dls_te, show_idx=mispredicted)
    logging.info('Validation f1 on epoch {} = {}'.format(epoch, validation_f1))
    # update target network weights
    TargetNetwork.update(DQNetwork)
    if validation_f1 > f1_ref:
      break


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='run_rgc.py', description='')
  argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
  argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
  argparser.add_argument('--key', metavar='KEY', default='test', type=str)
  argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
  argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  tfe.enable_eager_execution()

  main_execution(args.input, args.emb)
