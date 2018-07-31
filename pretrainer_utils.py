import logging
import numpy as np
import utility as u
import multiprocessing
import tensorflow as tf
from encoder import EncoderRNN
from decoder import DecoderRNN


def get_sequence_from_dddqn(dddqn, sos, lstm_state, max_steps, training=False, x=None):
  '''
  Retrieves predicted words from dddqn

  Inputs:
    -> dddqn, DDDQN instance
    -> sos, numpy array, batch of StartOfSequence token, shape = [batch_size, emb_dim]
    -> lstm_state, tuple of tensor (cell_state, hidden_state)
    -> max_steps, int, max number of tokens to produce
    -> training, boolean, optional, inference or training mode
    -> x, numpy array, optional, shape = [batch_size, num_tokens, emb_dim], must be given if training mode

  Outputs:
    -> predicted_words, tensor, shape = [batch_size, max_tokens]
    -> logits, tensor, shape = [batch_size, max_tokens, vocab_size]
  '''
  words = tf.convert_to_tensor(sos, dtype=tf.float32)
  x = np.asarray(x)
  predicted_words, logits = [], []
  for mt in range(max_steps):
    q, a, lstm_state, logit, words = dddqn.forward(words, lstm_state)
    if training:
      words = x[:,mt,:]
    predicted_words.append(a)
    logits.append(logit)
  predicted_words = tf.stack(predicted_words, axis=1)  # [max_steps, batch_size] to [batch_size, max_steps]
  logits = tf.stack(logits, axis=1)  # [num_steps, batch_size, vocab_size] to [batch_size, num_steps, vocab_size]
  return predicted_words, logits


def full_encoder_dddqn_pass(x, sl, encoder, dddqn, sos, max_steps, training=False):
  '''
  Performs a forward pass through the rgc ie through the encoder then dddqn

  Inputs:
    -> x,
    -> sl, list of int, list of last sequence indices
    -> encoder, EncoderRNN instance
    -> dddqn, DDDQN instance
    -> sos, numpy array, shape = [batch, emb_dim], Start of sequence token
    -> max_steps, int, the maximum number of tokens to produce
    -> training, boolean, optional, whether to trigger training mode or inference mode

  Outputs:
    -> preds, tensor, shape = [batch_size, time_step]
    -> logits, tensor, shape = [batch_size, max_tokens, vocab_size]
  '''
  output, cell_state = encoder.forward(x, sl)
  lstm_state = (cell_state, output)
  preds, logits = get_sequence_from_dddqn(dddqn, sos, lstm_state, max_steps, training=training, x=x)
  return preds, logits


def get_acc_full_dataset(dc, encoder, dddqn):
  '''
  Computes accuracy achieve on the whole dataset

  Inputs:
  -> dc, DataContainer instance
  -> encoder, EncoderRNN instance
  -> dddqn, DDDQN instance

  Outputs:
  -> acc, float, accuracy
  '''
  sos = dc.get_sos_batch_size(len(dc.x))
  preds, logits = full_encoder_dddqn_pass(dc.x, dc.sl, encoder, dddqn, sos, dc.max_tokens)
  _, acc = u.get_acc_word_seq(logits, dc.y_parrot_padded, dc.sl)
  return acc
