import os
import sys
import shutil
import logging
import numpy as np
import pickle as pk
import utility as u
import tensorflow as tf
from encoder import EncoderRNN
from decoder import DecoderRNN
import tensorflow.contrib.eager as tfe
from data_container import DataContainer
from pretrainer import see_parrot_results

import default


def get_data():
    dc = DataContainer(os.environ['INPUT'], os.environ['EMB'])
    x_a = [sample for batch in dc.x_train for sample in batch] + dc.x_te
    sl_a = [sample for batch in dc.sl_train for sample in batch] + dc.sl_te
    y_parrot_a = [sample for batch in dc.y_parrot_padded_batch for sample in batch] + dc.y_p_p_te
    sos = dc.get_sos_batch_size(len(x_a))
    encoder = EncoderRNN()
    decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=False)
    optimizer = tf.train.AdamOptimizer()
    x_batch = u.create_batch(x_a, batch_size=dc.batch_size)
    y_parrot_batch = u.create_batch(y_parrot_a, batch_size=dc.batch_size)
    sl_batch = u.create_batch(sl_a, batch_size=dc.batch_size)
    return dc, x_a, sl_a, y_parrot_a, sos, encoder, decoder, optimizer, x_batch, y_parrot_batch, sl_batch


def get_loss(encoder, decoder, epoch, x, y, sl, sos):
    output, cell_state = encoder.forward(x, sl)
    loss = decoder.get_loss(epoch, sos, (cell_state, output), y, sl, x, encoder.outputs)
    return loss


def test_saving_encoder_decoder():
    dc, x_a, sl_a, y_parrot_a, sos, encoder, decoder, optimizer, x_batch, y_parrot_batch, sl_batch = get_data()
    # three training pass encoder decoder
    logging.info('Train for 50 epochs...')
    logger = logging.getLogger()
    logger.disabled = True
    for epoch in range(50):
        for x, y, sl in zip(x_batch, y_parrot_batch, sl_batch):
            optimizer.minimize(lambda: get_loss(encoder, decoder, epoch, x, y, sl, dc.get_sos_batch_size(len(x))))
    logger.disabled = False
    see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos, greedy=True)

    grad_n_vars = optimizer.compute_gradients(lambda: get_loss(encoder, decoder, epoch, x, y, sl, dc.get_sos_batch_size(len(x))))
    vars = list(map(lambda x: x[1], grad_n_vars))
    print('trainables = ', [el.name for el in vars])
    # retrieve numpy values
    encoder_kernel, encoder_bias = list(map(lambda x: x.numpy(), encoder.encoder_cell.variables))
    decoder_lstm_kernel, decoder_lstm_bias = list(map(lambda x: x.numpy(), decoder.decoder_cell.variables))
    decoder_dense_kernel, decoder_dense_bias = list(map(lambda x: x.numpy(), decoder.word_predictor.variables))
    # save encoder decoder
    encoder.save('TMP/ENCODER')
    decoder.save('TMP/DECODER')
    # do another training pass encoder decoder
    optimizer.minimize(lambda: get_loss(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos))
    see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos, greedy=True)
    # retrieve new numpy values
    encoder_kernel_n1, encoder_bias_n1 = list(map(lambda x: x.numpy(), encoder.encoder_cell.variables))
    decoder_lstm_kernel_n1, decoder_lstm_bias_n1 = list(map(lambda x: x.numpy(), decoder.decoder_cell.variables))
    decoder_dense_kernel_n1, decoder_dense_bias_n1 = list(map(lambda x: x.numpy(), decoder.word_predictor.variables))
    # check that this new values are not equal to previous ones
    print('new encoder kernel = first kernel -> {}'.format(np.array_equal(encoder_kernel, encoder_kernel_n1)))
    print('new encoder bias = first bias -> {}'.format(np.array_equal(encoder_bias, encoder_bias_n1)))
    print('new decoder lstm kernel = first kernel -> {}'.format(np.array_equal(decoder_lstm_kernel, decoder_lstm_kernel_n1)))
    print('new decoder lstm bias = first bias -> {}'.format(np.array_equal(decoder_lstm_bias, decoder_lstm_bias_n1)))
    print('new decoder dense kernel = first kernel -> {}'.format(np.array_equal(decoder_dense_kernel, decoder_dense_kernel_n1)))
    print('new decoder dense bias = first bias -> {}\n\n'.format(np.array_equal(decoder_dense_bias, decoder_dense_bias_n1)))
    # load encoder decoder
    encoder.load('TMP/ENCODER')
    decoder.load('TMP/DECODER')
    see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos, greedy=True)
    # retrieve new values
    encoder_kernel_n2, encoder_bias_n2 = list(map(lambda x: x.numpy(), encoder.encoder_cell.variables))
    decoder_lstm_kernel_n2, decoder_lstm_bias_n2 = list(map(lambda x: x.numpy(), decoder.decoder_cell.variables))
    decoder_dense_kernel_n2, decoder_dense_bias_n2 = list(map(lambda x: x.numpy(), decoder.word_predictor.variables))
    # check that this new values are equal to first retrieved values
    print('loaded kernel = first kernel -> {}'.format(np.array_equal(encoder_kernel, encoder_kernel_n2)))
    print('loaded bias = first bias -> {}'.format(np.array_equal(encoder_bias, encoder_bias_n2)))
    print('loaded decoder lstm kernel = first kernel -> {}'.format(np.array_equal(decoder_lstm_kernel, decoder_lstm_kernel_n2)))
    print('loaded decoder lstm bias = first bias -> {}'.format(np.array_equal(decoder_lstm_bias, decoder_lstm_bias_n2)))
    print('loaded decoder dense kernel = first kernel -> {}'.format(np.array_equal(decoder_dense_kernel, decoder_dense_kernel_n2)))
    print('loaded decoder dense bias = first bias -> {}'.format(np.array_equal(decoder_dense_bias, decoder_dense_bias_n2)))


def test_reload():
    dc, x_a, sl_a, y_parrot_a, sos, encoder, decoder, optimizer, x_batch, y_parrot_batch, sl_batch = get_data()
    see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos, greedy=True)
    encoder.load('TMP/ENCODER')
    decoder.load('TMP/DECODER')
    see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, sos, greedy=True)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    rep = input('TEST saving encoder-decoder? (y or n): ')
    if rep == 'y':
        test_saving_encoder_decoder()
    rep = input('test reload? (y or n): ')
    if rep == 'y':
        test_reload()
    rep = input('remove TMP? (y or n): ')
    if rep == 'y':
        shutil.rmtree('models/TMP', ignore_errors=True)
