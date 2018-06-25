import os
import sys
import time
import tqdm
import heapq
import logging
import argparse
import numpy as np
import fasttext as ft
import multiprocessing
import tensorflow as tf
from collections import deque
from unidecode import unidecode
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import train_test_split

import utility as u

import default
sys.path.append(os.environ['LIBRARY'])
from library import DataLoader


def get_vocabulary(sources, emb, unidecode_lower=False):
    '''
    Gets vocabulary from the sources & creates word to index and index to word dictionaries

    Inputs:
        -> sources, list of string
        -> emb, fasttext embeddings
        -> unidecode_lower, boolean, whether or not to decode and lowerizing words

    Outputs:
        -> vocabulary, list of string
        -> word_to_idx, dictionary, map between word and index
        -> idx_to_word, dictionary, map between index and word
        -> idx_to_emb, dictionary, map between index and embedding representation
    '''
    if unidecode_lower:
        vocabulary = list(set([unidecode(w).lower() for s in sources for w in s.split(' ')]))
    else:
        vocabulary = list(set([w for s in sources for w in s.split(' ')]))
    logging.info('Vocabulary size = {}'.format(len(vocabulary)))
    word_to_idx = {w: vocabulary.index(w) for w in vocabulary}
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_emb = {v: emb[k] for k, v in word_to_idx.items()}
    return vocabulary, word_to_idx, idx_to_word, idx_to_emb


def get_y_parrot(sources, word2onehot, max_sl, pad_with='eos'):
    y_parrot, y_parrot_padded = [], []
    for s in sources:
        tmp = []
        for w in s.split(' '):
            tmp.append(word2onehot[w])
        y_parrot.append(tmp)

        tmp_pad = []
        for _ in range(max_sl - len(tmp)):
            tmp_pad.append(word2onehot[pad_with])
        y_parrot_padded.append(tmp + tmp_pad)
    return y_parrot, y_parrot_padded


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
        cleaned_sources = [u.clean_sentence(s) + ' eos' for s in self.data.sources]  # add EndOfSentence token
        decoded_lowered_sources = [unidecode(s).lower() for s in cleaned_sources]  # decode and lowerize sentences
        self.sos = np.vstack([self.emb['sos']] * self.batch_size)  # get embedding of SOS token and prepare it to batch size
        sources = u.to_emb(decoded_lowered_sources, self.emb)  # list of numpy array [num_tokens, embeddings_size]
        sources, seq_lengths, max_sl = u.pad_data(sources, pad_with=self.emb['eos'])
        labels, self.num_class = u.encode_labels(self.data.labels)
        self.max_tokens = max_sl

        # add StartOfSentence token to vocabulary
        self.vocabulary, self.word2idx, self.idx2word, self.idx2emb = get_vocabulary(decoded_lowered_sources + ['sos'], self.emb)
        self.word2onehot = u.one_hot_encoding(list(self.word2idx.keys()))
        self.y_parrot, self.y_parrot_padded = get_y_parrot(decoded_lowered_sources, self.word2onehot, self.max_tokens)

        # _tr = _train | _te = _test
        x_tr, self.x_te, y_tr, self.y_te, sl_tr, self.sl_te, y_p_tr, self.y_p_te, y_p_p_tr,\
        self.y_p_p_te = train_test_split(sources, labels, seq_lengths, self.y_parrot, self.y_parrot_padded,
                                         test_size=self.test_size, stratify=self.data.labels)

        sources, labels, seq_lengths, y_parrot_shuffled, y_p_p_s = u.shuffle_data(x_tr, y_tr, sl_tr, y_p_tr, y_p_p_tr)
        self.y_parrot_batch = u.create_batch(y_parrot_shuffled, batch_size=self.batch_size)
        self.y_parrot_padded_batch = u.create_batch(y_p_p_s, batch_size=self.batch_size)
        self.x_train, self.y_train, self.sl_train = u.to_batch(sources, labels, seq_lengths, batch_size=self.batch_size)

    def get_sos_batch_size(self, batch_size):
        return np.vstack([self.emb['sos']] * batch_size)

    def shuffle_training(self):
        self.x_train, self.y_train, self.sl_train = u.shuffle_data(self.x_train, self.y_train, self.sl_train)


class EncoderRNN(object):
    '''
    encoder_outputs: [max_time, batch_size, num_units] || encoder_state: [batch_size, num_units]
    '''
    def __init__(self, num_units=150, num_class=5, history_size=5):
        self.name = 'EncoderRNN'
        self.num_class = num_class
        self.num_units = num_units
        self.epoch = -1
        self.early_stoping = False
        self.validation_history = deque(maxlen=history_size)
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, name='encoder_lstm_cell')
        self.pred_layer_test = tf.layers.Dense(num_class, activation=None, name='encoder_classif_dense')

    def save(self, name=None):
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        saver = tfe.Saver(self.encoder_cell.variables + self.pred_layer_test.variables)
        saver.save(save_path)

    def load(self, name=None):
        self.classifier_predict(np.zeros((32, 16, 300), dtype=np.float32), list(range(2, 34, 1)))
        saver = tfe.Saver(self.encoder_cell.variables + self.pred_layer_test.variables)
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        saver.restore(save_path)

    def forward(self, x, sl, final=True, reverse=True):
        '''
        Performs a forward pass through a rnn

        Inputs:
            -> x, numpy array, shape = [batch_size, input_dim], example: [batch_size, sequence_length, embedding_dim]
            -> sl, list of int, list of last sequence indice for each sample in given batch
            -> final, boolean, optional, whether or not to return only the final output and cell state

        Outputs:
            -> final_output, numpy array, shape = [batch_size, cell_size]
        '''
        state = self.encoder_cell.zero_state(len(x), dtype=tf.float32)  # Initialize LSTM cell state with zeros
        unstacked_x = tf.unstack(x, axis=1)  # unstack the embeddings, shape = [time_steps, batch_size, emb_dim]
        if reverse:
            unstacked_x = reversed(unstacked_x)
        outputs, cell_states = [], []
        for input_step in unstacked_x:
            output, state = self.encoder_cell(input_step, state)  # state = (cell_state, hidden_state = output)
            outputs.append(output)
            cell_states.append(state[0])
        # outputs shape = [time_steps, batch_size, cell_size]
        outputs = tf.stack(outputs, axis=1)  # Stack outputs to (batch_size, time_steps, cell_size)
        cell_states = tf.stack(cell_states, axis=1)

        if final:
            if reverse:
                final_output = outputs[:,-1,:]
                final_cell_state = cell_states[:,-1,:]
            else:
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
            -> sl, list of int, list of last sequence indice for each sample

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
            -> sl, list of int, list of last sequence indice for each sample
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
            -> sl_test, list of int, list of last sequence indice for each sample

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
        return accuracy


def test_encoder(dataset, emb_path):
    '''
    Run classification test with the encoder model for sanity check
    '''
    dc = DataContainer(dataset, emb_path)
    optimizer = tf.train.AdamOptimizer()
    encoder = EncoderRNN(num_class=dc.num_class)
    save_path = 'models/' + encoder.name + '/'
    if os.path.isdir(save_path) and os.listdir(save_path):
        rep = input('Load saved encoder ? (y or n): ')
        if rep == 'y':
            encoder.load()
            encoder.validation('final', dc.x_te, dc.y_te, dc.sl_te)
    for epoch in range(300):
        for x, y, seq_length in zip(dc.x_train, dc.y_train, dc.sl_train):
            optimizer.minimize(lambda: encoder.get_loss(epoch, x, y, seq_length))
        encoder.validation(epoch, dc.x_te, dc.y_te, dc.sl_te)
        if encoder.early_stoping:
            break
        dc.shuffle_training()
    encoder.save()


class DecoderRNN(object):
    def __init__(self, word2idx, idx2word, idx2emb, num_units=150, max_tokens=128, beam_size=10):
        self.name = 'DecoderRNN'
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.beam_size = beam_size
        self.epoch = -1
        self.early_stoping = False
        self.parrot_stopping = False
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, name='decoder_lstm_cell')
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=tf.nn.relu, name='decoder_dense_wordpred')

    def save(self, name=None):
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        saver = tfe.Saver(self.decoder_cell.variables + self.word_predictor.variables)
        saver.save(save_path)

    def load(self, name=None):
        sos = np.zeros((32, 300), dtype=np.float32)
        state = self.decoder_cell.zero_state(32, dtype=tf.float32)
        self.forward(sos, state, [], list(range(2, 34, 1)))
        saver = tfe.Saver(self.decoder_cell.variables + self.word_predictor.variables)
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        saver.restore(save_path)

    def prediction(self, output, state):
        '''
        Performs a forward pass throught the network and returns the predicted probability
        for each word in the vocabulary

        Inputs:
            -> output, tensor, shape = [batch_size, emb_dim]
            -> state, (c, h)

        Outputs:
            -> logits, tensor, shape = [batch_size, vocabulary_size]
            -> state, (c, h)
        '''
        output, state = self.decoder_cell(output, state)
        logits = self.word_predictor(output)  # [batch_size, vocabulary_size]
        return tf.nn.softmax(logits), state

    def forward(self, sos, state, x, sl, training=False, greedy=False):
        '''
        Performs forward pass through the decoder network

        Inputs:
            -> sos, numpy array, batch of SOS token, shape = [batch_size, emb_dim]
            -> state, lstm_state_tuple (c, h)
            -> x, numpy array, shape = [batch_size, num_tokens, emb_dim]
            -> training, boolean, optional, default value to False

        Outputs:
            -> words_predicted, tensor
            -> words_logits, tensor
        '''
        #   INFERENCE BEHAVIOUR                  TRAINING BEHAVIOUR
        #  new_w1  new_w2  new_w3              new_w1  new_w2  new_w3
        #    ^       ^       ^                   ^       ^       ^
        #    |       |       |                   |       |       |
        #   LSTM    LSTM    LSTM                LSTM    LSTM    LSTM
        #    ^       ^       ^                   ^       ^       ^
        #    |       |       |                   |       |       |
        #   sos    new_w1  new_w2               sos      w1      w2
        #
        #   where w1, w2 are the word from input sentence
        x = np.asarray(x)
        if not training:
            if greedy:
                words_predicted = self.greedy_decoding(sos, state)
            else:
                # beam_search decoding with beam_size of 1 = greedy_decoding
                words_predicted = self.beam_search(sos, state)
            words_logits = None
        else:
            output = tf.convert_to_tensor(sos, dtype=tf.float32)
            words_predicted, words_logits = [], []
            for mt in range(self.max_tokens):
                pred_word, state, logits = self.greedy_forward(output, state)
                output = x[:,mt,:]
                words_predicted.append(pred_word)
                words_logits.append(logits)
            # [max_tokens, num_sample, vocab_size] to [num_samples, max_tokens, vocab_size]
            words_logits = tf.stack(words_logits, axis=1)
            # [max_tokens, num_sample] to [num_samples, max_tokens]
            words_predicted = tf.stack(words_predicted, axis=1)
        return words_predicted, words_logits

    def greedy_forward(self, output, state):
        '''
        Performs prediction then an argmax reduction on logits

        Inputs:
            -> output,
            -> state,

        Outputs:
            -> tensor, []
            -> state, (c, h)
            -> logits, tensor
        '''
        logits, state = self.prediction(output, state)
        return tf.argmax(logits, 1).numpy(), state, logits

    def greedy_decoding(self, output, state):
        '''
        Performs a greedy decoding on each given samples

        Inputs:
            -> output, list of np array, shape = [num_samples, emb_dim], [sos, sos, ...]
            -> state, (c, h), c_shape = [num_samples, num_units] = h_shape

        Outputs:
            -> words_predicted, tensor, shape = [num_samples, max_tokens]
        '''
        logging.info('Performs greedy decoding over given samples...')
        output = tf.convert_to_tensor(output, dtype=tf.float32)
        words_predicted = []
        for mt in range(self.max_tokens):
            pred_word, state, _ = self.greedy_forward(output, state)
            output = [self.i2e[i] for i in pred_word]
            words_predicted.append(pred_word)
        return tf.stack(words_predicted, axis=1)

    def beam_search(self, output, state):
        '''
        Performs beam search decoding on each given samples

        Inputs:
            -> output, list of np array, shape = [num_samples, emb_dim], [sos, sos, ...]
            -> state, (c, h), c_shape = [num_samples, num_units] = h_shape

        Outputs:
            -> words_predicted, tensor, shape = [num_samples, max_tokens]
        '''
        logging.info('Performs beam search decoding over given samples...')
        t = time.time()
        states = list(zip(tf.unstack(state[0]), tf.unstack(state[1])))
        words_predicted = []
        for sample, state in zip(output, states):
            state = (tf.reshape(state[0], [1, self.num_units]), tf.reshape(state[1], [1, self.num_units]))
            sample = tf.convert_to_tensor(sample.reshape((1, 300)), dtype=tf.float32)
            logits, state = self.prediction(sample, state)
            state = (tf.tile(state[0], [self.beam_size, 1]), tf.tile(state[1], [self.beam_size, 1]))
            values, indices = tf.nn.top_k(logits, k=self.beam_size)
            output = [self.i2e[w] for w in indices.numpy()[0]]
            tree_c, tree_w = values.numpy()[0], [[el] for el in indices.numpy()[0]]
            for i in range(self.max_tokens - 1):
                logits, state = self.prediction(output, state)
                values, indices = tf.nn.top_k(logits, k=self.beam_size)
                scores = np.ndarray.flatten(np.asarray([el + tree_c[j] for j, el in enumerate(values.numpy())]))
                words = np.ndarray.flatten(indices.numpy())
                ordered = heapq.nlargest(self.beam_size, range(len(scores)), scores.take)
                tree_c = scores[ordered]
                new_tree_w = [tree_w[(j-(j%self.beam_size))//self.beam_size] + [words[j]] for j in ordered]
                tree_w = new_tree_w
                output = [self.i2e[words[j]] for j in ordered]
            words_predicted.append(tree_w[0])
        logging.info('Beam search decoding time took = {}s'.format(round(time.time() - t, 3)))
        return tf.convert_to_tensor(words_predicted)

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
        eos_idx = {}
        for i in np.argwhere(full_sentence == self.w2i['eos']):
            if i[0] not in eos_idx:
                eos_idx[i[0]] = i[1]
        idxs_last_output = [eos_idx[i] if i in eos_idx else self.max_tokens - 1 for i in range(full_sentence.shape[0])]
        if pad:
            for s, i in zip(full_sentence, idxs_last_output):
                s[i+1:] = len(self.w2i) + 1
            final_full_sentence = tf.convert_to_tensor(full_sentence, dtype=tf.float32)
        else:
            final_full_sentence = [s[:i+1] for s, i in zip(full_sentence, idxs_last_output)]  # i+1 to include the EOS token
        return final_full_sentence

    def max_slice(self, logits, y):
        '''
        '''
        sliced_logits = []
        for i, s in enumerate(y):
            sliced_logits.append(logits[i,:len(s)+1,:])
        # return tf.convert_to_tensor(sliced_logits)
        return sliced_logits

    def cost(self, output, target, sl):
        '''
        Computes the cross entropy loss with respect to the given sequence lengths

        Inputs:
            -> output, tensor, the word logits, shape = [batch_size, num_tokens, emb_dim]
            -> target, tensor or numpy array, same shape as output
            -> sl, list of last sequence indice for each sample

        Outputs:
            -> float, the cross entropy loss
        '''
        # Compute cross entropy for each frame.
        # if we do not clip the value, it produces NAN
        cross_entropy = target * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
        cross_entropy = -tf.reduce_sum(cross_entropy, 2)
        mask = tf.cast(tf.sequence_mask(sl, output.shape[1]), dtype=tf.float32)
        # mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(cross_entropy)

    def get_loss(self, epoch, sos, state, y, sl, x, verbose=True):
        '''
        Computes loss from given batch

        Inputs:
            -> sos, numpy array, shape = [batch_size, emb_dim]
            -> state, lstm state tuple
            -> y, list of list of list

        Outputs:
            -> loss, float
        '''
        _, wl = self.forward(sos, state, x, sl, training=True)

        # +1 to sl because sl is a list of sequence length indices ie len(sequence) - 1
        sl = (np.asarray(sl) + 1).tolist()
        loss = self.cost(wl, y, sl)

        if verbose and epoch != self.epoch:
            predict = tf.cast(tf.argmax(wl, -1), dtype=tf.float32).numpy()
            target = np.argmax(y, -1)

            gp_word = 0
            gp_sentence = 0
            for p, t, size in zip(predict, target, sl):
                if np.array_equal(p[:size], t[:size]):
                    gp_sentence += 1
                gp_word += sum(np.equal(p[:size], t[:size]))

            acc_words = round(gp_word / sum(sl), 3)
            acc_sentences = round(gp_sentence / len(sl), 3)

            logging.info('Epoch {} -> loss = {} | acc_words = {} | acc_sentences = {}'.format(epoch, loss, acc_words, acc_sentences))
            self.epoch += 1

            if acc_sentences == 1.:
                self.parrot_stopping = True

        return loss

    def reconstruct_sentences(self, sentences):
        '''
        Reconstructs a sentence from a list of index

        Inputs:
            -> sentences, list of list, sublist = list of index

        Outputs:
            -> list of string
        '''
        return [' '.join([self.i2w[i] for i in s]) for s in sentences]


def pretrain_encoder(encoder, dc, idx=None, queue=None):
    '''
    Trains the encoder as a classifier

    Inputs:
        -> encoder, Encoder instance
        -> dc, DataContainer instance
    '''
    optimizer = tf.train.AdamOptimizer()
    for epoch in range(300):
        for x, y, seq_length in zip(dc.x_train, dc.y_train, dc.sl_train):
            optimizer.minimize(lambda: encoder.get_loss(epoch, x, y, seq_length))
        acc = encoder.validation(epoch, dc.x_te, dc.y_te, dc.sl_te)
        if encoder.early_stoping:
            break
        dc.shuffle_training()
    if idx is not None and queue:
        encoder.save(name='{}-{}'.format(encoder.name, idx))
        queue.put([acc, idx])


def choose_encoder(dc, search_size=8):
    '''
    Trains search_size encoder and return the best one
    '''
    procs = []
    queue = multiprocessing.Queue()
    encoder = EncoderRNN(num_class=dc.num_class)

    logging.info('Choosing encoder...')
    logger = logging.getLogger()
    logger.disabled = True

    for i in range(search_size):
        p = multiprocessing.Process(target=pretrain_encoder, args=(encoder, dc, i, queue))
        procs.append(p)
        p.start()

    results = []
    for i in range(len(procs)):
        results.append(queue.get())

    for process in procs:
        process.join()

    logger.disabled = False

    results.sort(key=lambda x: x[0], reverse=True)
    logging.info('Accuracy of the best encoder = {}'.format(results[0][0]))
    encoder.load(name='{}-{}'.format(encoder.name, results[0][1]))
    encoder.validation('final', dc.x_te, dc.y_te, dc.sl_te)
    return encoder


def parrot_initialization(dataset, emb_path):
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
        loss = decoder.get_loss(epoch, sos, (cell_state, output), y, sl, x)
        return loss

    def see_parrot_results(encoder, decoder, epoch, x, y, sl, sos, greedy=False):
        output, cell_state = encoder.forward(x, sl)
        wp, _ = decoder.forward(sos, (cell_state, output), x, sl, greedy=greedy)

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

    decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens)
    if os.path.isdir('models/Encoder-Decoder'):
        rep = input('Load previously trained Encoder-Decoder? (y or n): ')
        if rep == 'y':
            encoder = EncoderRNN(num_class=dc.num_class)
            encoder.load('Encoder-Decoder/Encoder')
            decoder.load('Encoder-Decoder/Decoder')
            see_parrot_results(encoder, decoder, 'final', x_a, y_parrot_a, sl_a, dc.get_sos_batch_size(len(x_a)))
            # ERROR, see_parrot_results doesn't dump the same acc with the loaded model than the saved model
            input()
        else:
            encoder = choose_encoder(dc)
    else:
        encoder = choose_encoder(dc, search_size=8)

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
        # strangely, shuffle data between epoch make the training noisy


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
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    # test_encoder(args.input, args.emb)
    parrot_initialization(args.input, args.emb)
