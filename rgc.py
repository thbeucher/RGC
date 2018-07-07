import os
import ast
import sys
import tqdm
import logging
import argparse
import numpy as np
import multiprocessing
import tensorflow as tf
from encoder import EncoderRNN
from decoder import DecoderRNN
from pretrainer import PreTrainer
import tensorflow.contrib.eager as tfe
from data_container import DataContainer

import default  # to import os environment variables
import utility as u


def pretrain_coder(coder, coder_cell, dc, idx=None, queue=None):
    '''
    Trains the encoder as a classifier

    Inputs:
        -> encoder, Encoder instance
        -> dc, DataContainer instance
    '''
    trainer = PreTrainer(dc.num_class, coder_cell)
    acc = trainer.classification_train([dc.x_train, dc.x_te], [dc.y_train, dc.y_te])
    if idx is not None and queue:
        coder.save(name='{}-{}'.format(coder.name, idx))
        queue.put([acc, idx])


def choose_coders(dc, attention, search_size=8):
    '''
    Trains search_size encoder and return the best one
    '''
    encoder = EncoderRNN()
    decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)

    logging.info('Choosing coders...')
    logger = logging.getLogger()
    logger.disabled = True

    def launch(coder, coder_cell, dc, search_size):
        procs = []
        queue = multiprocessing.Queue()

        for i in range(search_size):
            p = multiprocessing.Process(target=pretrain_coder, args=(coder, coder_cell, dc, i, queue))
            procs.append(p)
            p.start()

        results = []
        for i in range(len(procs)):
            results.append(queue.get())

        for process in procs:
            process.join()
        return results

    results_encoder = launch(encoder, encoder.encoder_cell, dc, search_size)
    results_decoder = launch(decoder, decoder.decoder_cell, dc, search_size)

    logger.disabled = False

    results_encoder.sort(key=lambda x: x[0], reverse=True)
    results_decoder.sort(key=lambda x: x[0], reverse=True)
    logging.info('Accuracy of the best encoder = {}'.format(results_encoder[0][0]))
    encoder.load(name='{}-{}'.format(encoder.name, results_encoder[0][1]))
    logging.info('Accuracy of the best decoder = {}'.format(results_decoder[0][0]))
    decoder.load(name='{}-{}'.format(decoder.name, results_decoder[0][1]), only_lstm=True)
    return encoder, decoder


def parrot_initialization(dataset, emb_path, attention):
    '''
    Trains the encoder-decoder to reproduce the input
    '''
    dc = DataContainer(dataset, emb_path)
    x_a = [sample for batch in dc.x_train for sample in batch] + dc.x_te
    sl_a = [sample for batch in dc.sl_train for sample in batch] + dc.sl_te
    y_parrot_a = [sample for batch in dc.y_parrot_padded_batch for sample in batch] + dc.y_p_p_te

    x_batch = u.create_batch(x_a, batch_size=dc.batch_size)
    y_parrot_batch = u.create_batch(y_parrot_a, batch_size=dc.batch_size)
    sl_batch = u.create_batch(sl_a, batch_size=dc.batch_size)

    def get_loss(encoder, decoder, epoch, x, y, sl, sos):
        output, cell_state = encoder.forward(x, sl)
        loss = decoder.get_loss(epoch, sos, (cell_state, output), y, sl, x, encoder.outputs)
        return loss

    def see_parrot_results(encoder, decoder, epoch, x, y, sl, sos, greedy=False):
        output, cell_state = encoder.forward(x, sl)
        wp, _ = decoder.forward(sos, (cell_state, output), x, sl, encoder.outputs, greedy=greedy)

        fwp = decoder.get_sequence(wp)
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

    if os.path.isdir('models/Encoder-Decoder'):
        rep = input('Load previously trained Encoder-Decoder? (y or n): ')
        if rep == 'y':
            encoder = EncoderRNN()
            decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)
            encoder.load('Encoder-Decoder/Encoder')
            decoder.load('Encoder-Decoder/Decoder')
            see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, dc.get_sos_batch_size(len(x_a)))
            # ERROR, see_parrot_results doesn't dump the same acc with the loaded model than the saved model
            input('ERR')
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
            see_parrot_results(encoder, decoder, epoch, x_a, y_parrot_a, sl_a, dc.get_sos_batch_size(len(x_a)), greedy=True)
            # see_parrot_results(encoder, decoder, epoch, x_a, y_parrot_a, sl_a, dc.get_sos_batch_size(len(x_a)))
        if decoder.parrot_stopping:
            break
        encoder.save(name='Encoder-Decoder/Encoder')
        decoder.save(name='Encoder-Decoder/Decoder')
        # x_batch, y_parrot_batch, sl_batch = u.shuffle_data(x_batch, y_parrot_batch, sl_batch)
        # strangely, shuffle data between epoch make the training realy noisy


if __name__ == '__main__':
    # # TODO:
    # => Add save & load of model
    #   -> currently, encoder save & load works but not Encoder-Decoder
    #
    # => Implement Attention mechanism (Badhanau or/and Luong)
    argparser = argparse.ArgumentParser(prog='rgc.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
    argparser.add_argument('--key', metavar='KEY', default='test', type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    parrot_initialization(args.input, args.emb, args.attention)
