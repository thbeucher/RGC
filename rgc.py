import os
import sys
import tqdm
import regex
import random
import logging
import argparse
import numpy as np
import fasttext as ft
import tensorflow as tf
from collections import deque
from unidecode import unidecode
import tensorflow.contrib.eager as tfe
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import default
sys.path.append(os.environ['LIBRARY'])
from library import DataLoader


def create_batch(iterable, batch_size=None, num_batch=None):
    '''
    Creates list of list
        if you provide batch_size, sublist will be of size batch_size
        if you provide num_batch, it will split your data into num_batch lists
    '''
    assert batch_size or num_batch, 'You have to provide batch_size or num_batch'
    size = len(iterable)
    if num_batch:
        assert num_batch < size
        p = size // num_batch
        return [iterable[i:i+p] for i in range(0, size, p)]
    else:
        return [iterable[i:i+batch_size] for i in range(0, size, batch_size)]


def clean_sentence(sentence):
    '''
    Removes double space and punctuation from given string
    '''
    return regex.sub(r' +', ' ', regex.sub(r'\p{Punct}', '', sentence)).strip()


def convert_to_emb(sentence, emb):
    '''
    Converts sentence to matrix with embedding representation of each tokens

    Inputs:
        -> sentence - string
        -> emb - fasttext embedding model

    Outpus:
        -> numpy ndim array - [number_of_token, embedding_dim]
    '''
    return np.asarray([emb[w] for w in sentence.split(' ')])


def to_emb(sentences, emb):
    '''
    Transforms a list of string into a list of numpy array

    Inputs:
        -> sentences - list of string
        -> emb_path - string - path to embeddings model
    '''
    logging.info('Transform sentences to embedding representation...')
    return [convert_to_emb(s, emb) for s in tqdm.tqdm(sentences)]


def pad_data(sources):
    '''
    Pads sequences to the same length defined by the biggest one (with value 0)

    Inputs:
        -> sources, list of numpy array of shape [sequence_length, embedding_dim]

    Outputs:
        -> sentences, list of numpy array, padded sequences
        -> sequence_lengths, list of int, list of legnth of each sequences
    '''
    logging.info('Padding sources...')
    sequence_lengths = [s.shape[0] - 1 for s in sources]
    max_seq_length = np.max(sequence_lengths) + 1
    sentences = [np.pad(s, pad_width=[(0, max_seq_length - s.shape[0]), (0, 0)], mode='constant', constant_values=0) for s in sources]
    return sentences, sequence_lengths


def encode_labels(labels):
    '''
    One hot encodings of labels

    Inputs:
        -> labels, list of string, list of labels

    Outputs:
        -> onehot_labels, numpy array of shape [num_samples, num_labels]
        -> num_labels, int, number of unique labels
    '''
    logging.info('Encoding labels...')
    uniq_labels = list(set(labels))
    onehot_labels = np.asarray([np.zeros(len(uniq_labels))] * len(labels))
    for i, l in enumerate(labels):
        onehot_labels[i][uniq_labels.index(l)] = 1
    return onehot_labels, len(uniq_labels)


def shuffle_data(*data):
    '''
    Shuffles data

    Inputs:
        -> give as many list as you want

    Outputs:
        -> shuffled lists
    '''
    logging.info('Shuffling data...')
    to_shuffle = list(zip(*data))
    random.shuffle(to_shuffle)
    return [list(map(lambda x: x[i], to_shuffle)) for i in range(len(data))]


def to_batch(sources, labels, sequence_lengths, batch_size):
    '''
    Transforms data into batchs

    Inputs:
        -> sources, list
        -> labels, list
        -> sequence_lengths, list
        -> batch_size, int

    Outputs:
        -> x_batch, list of list of size batch_size
        -> y_batch, list of list of size batch_size
        -> seq_length_batch, list of list of size batch_size
    '''
    logging.info('Transform data into batch...')
    x_batch = create_batch(sources, batch_size=batch_size)
    y_batch = create_batch(labels, batch_size=batch_size)
    seq_length_batch = create_batch(sequence_lengths, batch_size=batch_size)
    return x_batch, y_batch, seq_length_batch


def get_vocabulary(sources, emb):
    '''
    Gets vocabulary from the sources & creates word to index and index to word dictionaries

    Inputs:
        -> sources, list of string
    '''
    vocabulary = list(set([unidecode(w).lower() for s in sources for w in s.split(' ')]))
    logging.info('Vocabulary size = {}'.format(len(vocabulary)))
    word_to_idx = {w: vocabulary.index(w) for w in vocabulary}
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_emb = {v: emb[k] for k, v in word_to_idx.items()}
    return vocabulary, word_to_idx, idx_to_word, idx_to_emb

class DataContainer(object):
    def __init__(self, input_file, emb_path, batch_size=32, test_size=0.2):
        '''
        Inputs:
            -> input_file, string
            -> emb_path, string
            -> batch_size, int, optional, default value to 32
            -> test_size, float between 0 and 1, optional, proportion of the dataset to include to test split

        Avaialble data:
            -> x_batch, list of list of size batch_size
            -> y_batch, list of list of size batch_size
            -> seq_length_batch, list of list of size batch_size
            -> num_class, int, number of classes
            -> x_test, list of test source data
            -> y_test, numpy array, shape = [num_sample, num_class]
            -> sl_test, list of test sequence lengths data
            -> idx2emb, dictionary linking index to word embedding representation
            -> word2idx, dictionary linking word to index
            -> idx2word, dictionary linking index to word
            -> SOS, numpy array, shape = [batch_size, emb_dim], the Start Of Sentence token
            -> emb, fasttext model
        '''
        self.input_file = input_file
        self.emb_path = emb_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.prepare_data()

    def prepare_data(self):
        '''
        Loads & prepares training data into batches
        '''
        logging.info('Loads embeddings...')
        self.emb = ft.load_model(self.emb_path)
        logging.info('Embeddings loaded.')
        self.data = DataLoader(self.input_file)
        cleaned_sources = [clean_sentence(s) + ' EOS' for s in self.data.sources]  # add EndOfSentence token
        self.sos = np.vstack([self.emb['sos']] * self.batch_size)  # get embedding of SOS token and prepare it to batch size
        sources = to_emb(cleaned_sources, self.emb)  # list of numpy array [num_tokens, embeddings_size]
        sources, seq_lengths = pad_data(sources)
        labels, self.num_class = encode_labels(self.data.labels)

        # add StartOfSentence token to vocabulary
        self.vocabulary, self.word2idx, self.idx2word, self.idx2emb = get_vocabulary(cleaned_sources + ['sos'], self.emb)
        self.y_parrot = [[self.word2idx[w] for w in unidecode(s).lower().split(' ')] for s in cleaned_sources]

        # _tr = _train | _te = _test
        x_tr, self.x_te, y_tr, self.y_te, sl_tr, self.sl_te, y_p_tr, self.y_p_te = train_test_split(sources, labels, seq_lengths,
                                                                                               self.y_parrot,
                                                                                               test_size=self.test_size,
                                                                                               stratify=self.data.labels)

        sources, labels, seq_lengths, y_parrot_shuffled = shuffle_data(x_tr, y_tr, sl_tr, y_p_tr)
        self.y_parrot_batch = create_batch(y_parrot_shuffled, batch_size=self.batch_size)
        self.x_train, self.y_train, self.sl_train = to_batch(sources, labels, seq_lengths, self.batch_size)


class EncoderRNN(object):
    '''
    encoder_outputs: [max_time, batch_size, num_units] || encoder_state: [batch_size, num_units]
    '''
    def __init__(self, num_units=150, num_class=5, history_size=5):
        self.num_class = num_class
        self.num_units = num_units
        self.epoch = -1
        self.early_stoping = False
        self.validation_history = deque(maxlen=history_size)
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        self.pred_layer_test = tf.layers.Dense(num_class, activation=None)

    def forward(self, x, sl, final=True):
        '''
        Performs a forward pass through a rnn

        Inputs:
            -> x, numpy array, shape = [batch_size, input_dim], example: [batch_size, sequence_length, embedding_dim]
            -> sl, list of int, the list of sequence length for each sample in given batch
            -> final, boolean, optional, whether or not to return only the final output and cell state

        Outputs:
            -> final_output, numpy array, shape = [batch_size, cell_size]
        '''
        state = self.encoder_cell.zero_state(len(x), dtype=tf.float32)  # Initialize LSTM cell state with zeros
        unstacked_x = tf.unstack(x, axis=1)  # unstack the embeddings, shape = [time_steps, batch_size, emb_dim]
        outputs, cell_states = [], []
        for input_step in unstacked_x:
            output, state = self.encoder_cell(input_step, state)  # state = (cell_state, hidden_state = output)
            outputs.append(output)
            cell_states.append(state[0])
        # outputs shape = [time_steps, batch_size, cell_size]
        outputs = tf.stack(outputs, axis=1)  # Stack outputs to (batch_size, time_steps, cell_size)
        cell_states = tf.stack(cell_states, axis=1)

        if final:
            idxs_last_output = tf.stack([tf.range(len(x)), sl], axis=1)  # get end index of each sequence
            final_output = tf.gather_nd(outputs, idxs_last_output)  # retrieve last output for each sequence
            final_cell_state = tf.gather_nd(cell_states, idxs_last_output)
            return final_output, final_cell_state
        else:
            return outputs, cell_states

    def classifier_predict(self, x, sl):
        '''
        Performs a forward pass then go through a prediction layer to get predicted classes

        Inputs:
            -> x, numpy array, shape = [batch_size, input_dim]
            -> sl, list of int, list of sequence length

        Outputs:
            -> logits, tensor, shape = [batch_size, num_class]
        '''
        final_output, final_cell_state = self.forward(x, sl)
        # dropped_output = tf.layers.dropout(final_output, rate=0.3, training=True)  # regularization
        logits = self.pred_layer_test(final_output)
        return logits

    def get_loss(self, epoch, x, y, sl, verbose=True):
        '''
        Computes the loss from a forward pass

        Inputs:
            -> epoch, int
            -> x, numpy array, shape = [batch_size, input_dim]
            -> y, numpy array, shape = [batch_size, num_class]
            -> sl, list of int, list of sequence length
            -> verbose, boolean, decide wether or not print the loss & accuracy at each epoch
        '''
        logits = self.classifier_predict(x, sl)

        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        y_loss = [el.tolist().index(1) for el in y]  # if using sparse_softmax_cross_entropy_with_logits
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_loss, logits=logits)

        loss = tf.reduce_mean(loss)

        if verbose and epoch != self.epoch:
            predict = tf.argmax(logits, 1).numpy()
            target = np.argmax(y, 1)
            accuracy = np.sum(predict == target) / len(target)
            logging.info('Epoch {} -> loss = {} | acc = {}'.format(epoch, loss, round(accuracy, 3)))
            self.epoch += 1

        return loss

    def validation(self, epoch, x_test, y_test, sl_test):
        '''
        Computes validation accuracy and define an early stoping

        Inputs:
            -> epoch, int
            -> x_test, numpy array, shape = [num_test_samples, input_dim]
            -> y_test, numpy array, shape = [num_test_samples, num_class]
            -> sl_test, list of int, list of sequence length

        Outputs:
            Print validation accuracy and set a boolean (early_stoping) to True
            if stoping criterias are filled

            Stoping criterias:
                -> new validation accuracy is lower than all accuracies stored in validation_history
                -> new accuracy is equal to all accuracies stored in validation_history
        '''
        logits = self.classifier_predict(x_test, sl_test)
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


def unitest_encoder(dataset, emb_path):
    '''
    Run classification test with the encoder model for sanity check
    '''
    dc = DataContainer(dataset, emb_path)
    optimizer = tf.train.AdamOptimizer()
    encoder = EncoderRNN(num_class=dc.num_class)
    for epoch in range(300):
        for x, y, seq_length in zip(dc.x_train, dc.y_train, dc.sl_train):
            optimizer.minimize(lambda: encoder.get_loss(epoch, x, y, seq_length))
        encoder.validation(epoch, dc.x_te, dc.y_te, dc.sl_te)
        if encoder.early_stoping:
            break


class DecoderRNN(object):
    def __init__(self, word2idx, idx2word, idx2emb, num_units=150, max_tokens=128):
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.epoch = -1
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=None)

    def forward(self, sos, state):
        '''
        Performs forward pass through the decoder network

        Inputs:
            -> sos, numpy array, batch of SOS token, shape = [batch_size, emb_dim]
            -> state, lstm_state_tuple (c, h)

        Outputs:
            -> words_predicted, tensor
            -> words_logits, tensor
        '''
        output = tf.convert_to_tensor(sos, dtype=tf.float32)
        words_predicted = []
        words_logits = []
        for _ in range(self.max_tokens):
            output, state = self.decoder_cell(output, state)
            logits = self.word_predictor(output)
            pred_word = tf.argmax(logits, 1).numpy()
            output = [self.i2e[i] for i in pred_word]
            words_predicted.append(pred_word)
            words_logits.append(logits)
        words_predicted = tf.stack(words_predicted, axis=1)  # stack output to [batch_size, max_tokens]
        words_logits = tf.stack(words_logits, axis=1)  # stack output to [batch_size, max_tokensa]
        return words_predicted, words_logits

    def get_sequence(self, full_sentence, pad=False):
        '''
        Slices samples in order to stop at EOS token

        Inputs:
            -> full_sentence, tensor, shape = [batch_size, max_tokens]
            -> pad, boolean, optional, whether or not to return a list of tensor
               of shape sequence_length or to return a tensor of shape [batch_size, max_tokens]
               where tokens after EOS token are padded to len(w2i) + 1

        Outputs:
            -> final_full_sentence:
                -> list of tensor, each tensor of shape = [sequence_length]
                or
                -> tensor, shape = [batch_size, max_tokens]
        '''
        full_sentence = full_sentence.numpy()
        full_sentence[0][100] = self.w2i['eos']
        full_sentence[5][115] = self.w2i['eos']
        eos_idx = {i[0]: i[1] for i in np.argwhere(full_sentence == self.w2i['eos'])}  # find EOS indices
        idxs_last_output = [eos_idx[i] if i in eos_idx else self.max_tokens - 1 for i in range(full_sentence.shape[0])]
        if pad:
            for s, i in zip(full_sentence, idxs_last_output):
                s[i+1:] = len(self.w2i) + 1
            final_full_sentence = tf.convert_to_tensor(full_sentence)
        else:
            final_full_sentence = [s[:i+1] for s, i in zip(full_sentence, idxs_last_output)]  # i+1 to include the EOS token
        return final_full_sentence

    def get_loss(self, x, y):
        '''

        Inputs:
            -> x,
            -> y, list of list

        Outputs:
            ->
        '''
        wp, wl = self.forward(x, self.decoder_cell.zero_state(32, dtype=tf.float32))
        fwp = self.get_sequence(wp, pad=True)
        y_pad = tf.convert_to_tensor([s + [len(self.w2i) + 1] * (self.max_tokens - len(s)) for s in y])
        print(y_pad.shape)
        input(fwp.shape)
        # np.pad(s, pad_width=[(0, max_seq_length - s.shape[0]), (0, 0)], mode='constant', constant_values=0)
        # tf.nn.sigmoid_cross_entropy_with_logits(labels=, logits=)

    def test(self, x, y):
        self.get_loss(x, y)


def parrot_initialization(dataset, emb_path):
    '''
    Trains the encoder-decoder to reproduce the input
    '''
    dc = DataContainer(dataset, emb_path)
    x = [sample for batch in dc.x_train for sample in batch] + dc.x_te
    y = np.vstack([np.asarray([sample for batch in dc.y_train for sample in batch]), dc.y_te])
    y_parrot = [sample for batch in dc.y_parrot_batch for sample in batch] + dc.y_p_te
    x_batch, y_parrot_batch = create_batch(x, dc.batch_size), create_batch(y_parrot, dc.batch_size)
    encoder = EncoderRNN(num_class=dc.num_class)
    decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb)
    decoder.test(dc.sos, y_parrot_batch[0])
    input()
    optimizer = tf.train.AdamOptimizer()
    for epoch in range(300):
        for x, y, seq_length in zip(dc.x_train, dc.y_train, dc.sl_train):
            optimizer.minimize(lambda: decoder.get_loss(epoch, x, y, seq_length))
        encoder.validation(epoch, dc.x_te, dc.y_te, dc.sl_te)
        if encoder.early_stoping:
            break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='rgc.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
    argparser.add_argument('--key', metavar='KEY', default='test', type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    # unitest_encoder(args.input, args.emb)
    parrot_initialization(args.input, args.emb)
