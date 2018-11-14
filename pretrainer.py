import os
import logging
import numpy as np
import utility as u
import tensorflow as tf
from dddqn import DDDQN
from collections import deque
import pretrainer_utils as pu
from encoder import EncoderRNN
from decoder import DecoderRNN
from data_container import DataContainer


class RNNPreTrainer(object):
  def __init__(self, num_class, layer_to_train, history_size=5):
    self.num_class = num_class
    self.layer_to_train = layer_to_train
    self.early_stoping = False
    self.epoch = -1
    self.validation_history = deque(maxlen=history_size)
    self.optimizer = tf.train.AdamOptimizer()
    self.prediction_layer = tf.layers.Dense(num_class, activation=None, name='prediction_layer')

  def forward(self, x):
    '''
    Performs a forward pass then go through a prediction layer to get predicted classes

    Inputs:
      -> x, numpy array, shape = [batch_size, input_dim]

    Outputs:
      -> logits, tensor, shape = [batch_size, num_class]
    '''
    state = self.layer_to_train.zero_state(len(x), dtype=tf.float64)  # Initialize LSTM cell state with zeros
    unstacked_x = tf.unstack(x, axis=1)  # unstack the embeddings, shape = [time_steps, batch_size, emb_dim]
    unstacked_x = reversed(unstacked_x)
    outputs = []
    for input_step in unstacked_x:
      output, state = self.layer_to_train(input_step, state)  # state = (cell_state, hidden_state = output)
      outputs.append(output)
    outputs = tf.stack(outputs, axis=1)  # reshape from [times, batch_size, num_units] to [batch_size, times, num_units]
    logits = self.prediction_layer(output)
    return logits

  def get_loss(self, epoch, x, y, verbose=True):
    '''
    Computes the loss from a forward pass

    Inputs:
      -> epoch, int
      -> x, numpy array, shape = [batch_size, input_dim]
      -> y, numpy array, shape = [batch_size, num_class]
      -> verbose, boolean, decide wether or not print the loss & accuracy at each epoch
    '''
    y_loss = [el.tolist().index(1) for el in y]  # if using sparse_softmax_cross_entropy_with_logits
    logits = self.forward(x)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_loss, logits=logits)
    loss = tf.reduce_mean(loss)

    if verbose and epoch != self.epoch:
      accuracy = u.get_accuracy(logits, y)
      logging.info('Epoch {} -> loss = {} | acc = {}'.format(epoch, loss, accuracy))
      self.epoch += 1

    return loss

  def classification_train(self, x, y):
    '''
      x = [x_train, x_test]
      y = [y_train, y_test]
    '''
    x_train, x_test = x
    y_train, y_test = y
    for epoch in range(300):
      for x, y in zip(x_train, y_train):
        self.optimizer.minimize(lambda: self.get_loss(epoch, x, y))
      acc = self.validation(epoch, x_test, y_test)
      if self.early_stoping:
        break
    return acc

  def validation(self, epoch, x_test, y_test):
    '''
    Computes validation accuracy and define an early stoping

    Inputs:
      -> epoch, int
      -> x_test, numpy array, shape = [num_test_samples, input_dim]
      -> y_test, numpy array, shape = [num_test_samples, num_class]

    Outputs:
      Print validation accuracy and set a boolean (early_stoping) to True
      if stoping criterias are filled

      Stoping criterias:
        -> new validation accuracy is lower than all accuracies stored in validation_history
        -> new accuracy is equal to all accuracies stored in validation_history
    '''
    logits = self.forward(x_test)
    predictions = tf.argmax(logits, 1).numpy()
    targets = np.argmax(y_test, 1)
    accuracy = round(np.sum(predictions == targets) / len(targets), 3)

    logging.info('Epoch {} -> Validation accuracy score = {}'.format(epoch, accuracy))
    preliminary_test = len(self.validation_history) == self.validation_history.maxlen
    test_decrease = all(accuracy < acc for acc in self.validation_history)
    test_equal = all(accuracy == acc for acc in self.validation_history)

    if preliminary_test and (test_decrease or test_equal):
      logging.info('Early stoping criteria raised: {}'.format('decreasing' if test_decrease else 'plateau'))
      self.early_stoping = True
    else:
      self.validation_history.append(accuracy)

    return accuracy


def pretrain_rnn_layer(class_rnn, layer_to_train, dc, idx, queue, multiprocessed=True):
  '''
  Trains the given rnn layer using classification task

  Inputs:
    -> class_rnn, class object, should contain a name attribute and a save function
    -> layer_to_train, the tensorflow layer to pretrain
    -> dc, DataContainer instance
    -> idx, int, index
    -> queue, multiprocessing Queue object
  '''
  trainer = RNNPreTrainer(dc.num_class, layer_to_train)
  acc = trainer.classification_train([dc.x_train, dc.x_te], [dc.y_train_classif, dc.y_te_classif])
  class_rnn.save(name='{}-{}'.format(class_rnn.name, idx))
  if multiprocessed:
    queue.put([acc, idx])
  else:
    return [acc, idx]


def choose_best_rnn_pretrained(class_rnn, layer_to_train, dc, search_size=8, log=False, multiprocessed=False):
  '''
  Trains a pool of classifier and retrieve the best one to initialize the weights of the rnn layer
  '''
  logging.info('Choosing best rnn layer pretrained...')
  if not log:
    logger = logging.getLogger()
    logger.disabled = True

  if multiprocessed:
    results = u.multiples_launch(pretrain_rnn_layer, [class_rnn, layer_to_train, dc], num_process=search_size)
  else:
    results = [pretrain_rnn_layer(class_rnn, layer_to_train, dc, i, None, multiprocessed=False) for i in range(search_size)]

  if not log:
    logger.disabled = False

  results.sort(key=lambda x: x[0], reverse=True)
  logging.info('Accuracy of the best classifier = {}'.format(results[0][0]))
  class_rnn.load(name='{}-{}'.format(class_rnn.name, results[0][1]), only_lstm=True)


def choose_coders(dc, attention, search_size=8):
  '''
  Trains search_size coders and return the best one
  '''
  encoder = EncoderRNN()
  decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)

  logging.info('Choosing coders...')
  logger = logging.getLogger()
  logger.disabled = True

  results_encoder = u.multiples_launch(pretrain_rnn_layer, [encoder, encoder.encoder_cell, dc], num_process=search_size)
  results_decoder = u.multiples_launch(pretrain_rnn_layer, [decoder, decoder.decoder_cell, dc], num_process=search_size)

  logger.disabled = False

  results_encoder.sort(key=lambda x: x[0], reverse=True)
  results_decoder.sort(key=lambda x: x[0], reverse=True)
  logging.info('Accuracy of the best encoder = {}'.format(results_encoder[0][0]))
  encoder.load(name='{}-{}'.format(encoder.name, results_encoder[0][1]))
  logging.info('Accuracy of the best decoder = {}'.format(results_decoder[0][0]))
  decoder.load(name='{}-{}'.format(decoder.name, results_decoder[0][1]), only_lstm=True)
  return encoder, decoder


def see_parrot_results(encoder, decoder, epoch, x, y, sl, sos, greedy=False):
  output, cell_state = encoder.forward(x, sl)
  wp, _ = decoder.forward(sos, (cell_state, output), x, sl, encoder.outputs, greedy=greedy)

  y_idx = np.argmax(y, axis=-1)
  target = [s[:size+1] for s, size in zip(y_idx, sl)]
  target_sentences = decoder.reconstruct_sentences(target)

  predict = decoder.get_sequence(wp)
  predict_sentences = decoder.reconstruct_sentences(predict)

  acc = sum([t == p for t, p in zip(target_sentences, predict_sentences)]) / len(target_sentences)
  logging.info('Accuracy on all sentences = {}'.format(round(acc, 3)))

  with open('parrot_results_extract.txt', 'a') as f:
    f.write('Epoch {}:\n'.format(epoch))
    for t, p in zip(target_sentences[:10], predict_sentences[:10]):
      f.write('Target -> ' + t + '\nPred -> ' + p + '\n\n')
    f.write('\n\n\n\n')


def parrot_initialization_encoder_decoder(dataset, emb_path, attention):
  '''
  Trains the encoder-decoder to reproduce the input
  '''
  dc = DataContainer(dataset, emb_path)
  dc.prepare_data()

  x_batch, y_parrot_batch, sl_batch = u.to_batch(dc.x, dc.y_parrot_padded, dc.sl, batch_size=dc.batch_size)

  def get_loss(encoder, decoder, epoch, x, y, sl, sos):
    output, cell_state = encoder.forward(x, sl)
    loss = decoder.get_loss(epoch, sos, (cell_state, output), y, sl, x, encoder.outputs)
    return loss

  if os.path.isdir('models/Encoder-Decoder'):
    rep = input('Load previously trained Encoder-Decoder? (y or n): ')
    if rep == 'y' or rep == '':
      encoder = EncoderRNN()
      decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)
      encoder.load(name='Encoder-Decoder/Encoder')
      decoder.load(name='Encoder-Decoder/Decoder')
      sos = dc.get_sos_batch_size(len(dc.x))
      see_parrot_results(encoder, decoder, 'final', dc.x, dc.y_parrot_padded, dc.sl, sos, greedy=True)
    else:
      encoder, decoder = choose_coders(dc, attention, search_size=5)
  else:
    encoder, decoder = choose_coders(dc, attention, search_size=5)

  optimizer = tf.train.AdamOptimizer()

  for epoch in range(300):
    for x, y, sl in zip(x_batch, y_parrot_batch, sl_batch):
      sos = dc.get_sos_batch_size(len(x))
      # grad_n_vars = optimizer.compute_gradients(lambda: get_loss(encoder, decoder, epoch, x, y, sl, sos))
      # optimizer.apply_gradients(grad_n_vars)
      optimizer.minimize(lambda: get_loss(encoder, decoder, epoch, x, y, sl, sos))
    if epoch % 30 == 0:
      # to reduce training time, compute global accuracy only every 30 epochs
      sos = dc.get_sos_batch_size(len(dc.x))
      see_parrot_results(encoder, decoder, epoch, dc.x, dc.y_parrot_padded, dc.sl, sos, greedy=True)
      # see_parrot_results(encoder, decoder, epoch, dc.x, dc.y_parrot_padded, dc.sl, sos)
    encoder.save(name='Encoder-Decoder/Encoder')
    decoder.save(name='Encoder-Decoder/Decoder')
    if decoder.parrot_stopping:
      break
    # x_batch, y_parrot_batch, sl_batch = u.shuffle_data(x_batch, y_parrot_batch, sl_batch)
    # strangely, shuffle data between epoch make the training realy noisy

  return encoder, decoder, dc


def parrot_initialization_rgc(dataset, emb_path, dc=None, encoder=None, dddqn=None):
  '''
  Trains the rgc to repeat the input
  '''
  # TODO save optimizer
  if dc is None:
    dc = DataContainer(dataset, emb_path)
    dc.prepare_data()
  x_batch, y_parrot_batch, sl_batch = u.to_batch(dc.x, dc.y_parrot_padded, dc.sl, batch_size=dc.batch_size)

  # initialize rnn cell of the encoder and the dddqn
  rep = input('Load RNN cell pretrained for the encoder & dddqn? (y or n): ')
  if encoder is None:
    encoder = EncoderRNN(num_units=256)
  if rep == 'y' or rep == '':
    encoder.load(name='EncoderRNN-0')
  else:
    choose_best_rnn_pretrained(encoder, encoder.encoder_cell, dc, search_size=1, multiprocessed=False)
  # we do not need to train the dddqn rnn layer since we already trained the encoder rnn layer
  # we just have to initialize the dddqn rnn layer weights with the ones from the encoder
  if dddqn is None:
    dddqn = DDDQN(dc.word2idx, dc.idx2word, dc.idx2emb)
  u.init_rnn_layer(dddqn.lstm)
  u.update_layer(dddqn.lstm, encoder.encoder_cell)

  # define the loss function used to pretrain the rgc
  def get_loss(encoder, dddqn, epoch, x, y, sl, sos, max_steps, verbose=True):
    preds, logits, _, _, _ = pu.full_encoder_dddqn_pass(x, sl, encoder, dddqn, sos, max_steps, training=True)
    logits = tf.nn.softmax(logits)  # normalize logits between 0 & 1 to allow training through cross-entropy
    sl = [end_idx + 1 for end_idx in sl]  # sl = [len(sequence)-1, ...] => +1 to get the len
    loss = u.cross_entropy_cost(logits, y, sequence_lengths=sl)
    if verbose:
      acc_words, acc_sentences = u.get_acc_word_seq(logits, y, sl)
      logging.info('Epoch {} -> loss = {} | acc_words = {} | acc_sentences = {}'.format(epoch, loss, acc_words, acc_sentences))
    return loss

  rep = input('Load pretrained RGC-ENCODER-DDDQN? (y or n): ')
  if rep == 'y' or rep == '':
    encoder.load('RGC/Encoder')
    dddqn.load('RGC/DDDQN')

  rep = input('Train RGC-ENCODER-DDDQN? (y or n): ')
  if rep == 'y' or rep == '':
    optimizer = tf.train.AdamOptimizer()
    # training loop over epoch and batchs
    for epoch in range(300):
      verbose = True
      for x, y, sl in zip(x_batch, y_parrot_batch, sl_batch):
        sos = dc.get_sos_batch_size(len(x))
        optimizer.minimize(lambda: get_loss(encoder, dddqn, epoch, x, y, sl, sos, dc.max_tokens, verbose=verbose))
        verbose = False
      encoder.save(name='RGC/Encoder')
      dddqn.save(name='RGC/DDDQN')
      acc = pu.get_acc_full_dataset(dc, encoder, dddqn)
      logging.info('Validation accuracy = {}'.format(acc))
      if acc > 0.95:
        logging.info('Stopping criteria on validation accuracy raised')
        break

  return encoder, dddqn, dc


if __name__ == '__main__':
  import os
  import sys
  import default
  import tensorflow.contrib.eager as tfe
  from data_container import DataContainer

  logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  tfe.enable_eager_execution()

  parrot_initialization_rgc(os.environ['INPUT'], os.environ['EMB'])

  # dc = DataContainer(os.environ['INPUT'], os.environ['EMB'])
  # dc.prepare_data()

  # encoder = EncoderRNN()
  # trainer = RNNPreTrainer(dc.num_class, encoder.encoder_cell)
  # acc = trainer.classification_train([dc.x_train, dc.x_te], [dc.y_train_classif, dc.y_te])
