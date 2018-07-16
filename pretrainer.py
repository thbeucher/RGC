import os
import logging
import numpy as np
import utility as u
import multiprocessing
import tensorflow as tf
from collections import deque
from encoder import EncoderRNN
from decoder import DecoderRNN
from data_container import DataContainer


class PreTrainer(object):
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
        state = self.layer_to_train.zero_state(len(x), dtype=tf.float32)  # Initialize LSTM cell state with zeros
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
            predict = tf.argmax(logits, 1).numpy()
            target = np.argmax(y, 1)
            accuracy = np.sum(predict == target) / len(target)
            logging.info('Epoch {} -> loss = {} | acc = {}'.format(epoch, loss, round(accuracy, 3)))
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


def pretrain_coder(coder, coder_cell, dc, idx=None, queue=None):
    '''
    Trains the encoder as a classifier

    Inputs:
        -> encoder, Encoder instance
        -> dc, DataContainer instance
    '''
    trainer = PreTrainer(dc.num_class, coder_cell)
    acc = trainer.classification_train([dc.x_train, dc.x_te], [dc.y_train_classif, dc.y_te_classif])
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


def parrot_initialization(dataset, emb_path, attention):
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
        if rep == 'y':
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


if __name__ == '__main__':
    import os
    import sys
    import default
    from encoder import EncoderRNN
    import tensorflow.contrib.eager as tfe
    from data_container import DataContainer

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    dc = DataContainer(os.environ['INPUT'], os.environ['EMB'])
    dc.prepare_data()
    encoder = EncoderRNN()
    trainer = PreTrainer(dc.num_class, encoder.encoder_cell)

    acc = trainer.classification_train([dc.x_train, dc.x_te], [dc.y_train_classif, dc.y_te])
