import os
import heapq
import logging
import numpy as np
import utility as u
from time import time
import tensorflow as tf
from attention import AttentionS2S
import tensorflow.contrib.eager as tfe


# # TODO:
# => Luong attention implemented but the attention vector is not feed to the decoder input
class DecoderRNN(object):
    def __init__(self, word2idx, idx2word, idx2emb, num_units=150, max_tokens=128, beam_size=10, attention=False):
        self.name = 'DecoderRNN'
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.beam_size = beam_size
        self.attention = attention
        self.epoch = -1
        self.early_stoping = False
        self.parrot_stopping = False
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, name='decoder_lstm_cell')
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=None, name='decoder_dense_wordpred')
        if attention:
            logging.info('Attention mechanism activated for the decoder...')
            self.attn = AttentionS2S(num_units)

    def save(self, name=None):
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        saver = tfe.Saver(self.decoder_cell.variables + self.word_predictor.variables)
        saver.save(save_path)

    def load(self, name=None, only_lstm=False):
        x = [np.zeros((25, 300))] * 32
        sos = np.zeros((32, 300), dtype=np.float64)
        state = self.decoder_cell.zero_state(32, dtype=tf.float64)
        outputs = np.zeros((32, 25, 150), dtype=np.float64)
        self.forward(sos, state, x, list(range(2, 34, 1)), outputs, training=True)
        if only_lstm:
            saver = tfe.Saver(self.decoder_cell.variables)
        else:
            saver = tfe.Saver(self.decoder_cell.variables + self.word_predictor.variables)
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        saver.restore(save_path)

    def prediction(self, output, state):
        '''
        Performs a forward pass throught the network and returns the predicted probability
        for each word in the vocabulary

        Inputs:
            -> output, tensor, shape = [batch_size, cell_size]
            -> state, (c, h)

        Outputs:
            -> logits, tensor, shape = [batch_size, vocabulary_size]
            -> state, (c, h)
        '''
        output, state = self.decoder_cell(output, state)

        if self.attention:
            self.attn.forward(output, self.encoder_outputs)

        logits = self.word_predictor(output)  # [batch_size, vocabulary_size]
        return tf.nn.softmax(logits), state

    def forward(self, sos, state, x, sl, encoder_outputs, training=False, greedy=False):
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
        self.encoder_outputs = encoder_outputs
        x = np.asarray(x)
        if not training:
            if greedy:
                words_predicted = self.greedy_decoding(sos, state)
            else:
                words_predicted = self.beam_search(sos, state)  # beam_search decoding with beam_size of 1 = greedy_decoding
            words_logits = None
        else:
            output = tf.convert_to_tensor(sos, dtype=tf.float64)
            words_predicted, words_logits = [], []
            for mt in range(self.max_tokens):
                pred_word, state, logits = self.greedy_forward(output, state)
                output = x[:,mt,:]
                words_predicted.append(pred_word)
                words_logits.append(logits)
            # [max_tokens, num_sample, vocab_size] to [num_samples, max_tokens, vocab_size]
            words_logits = tf.stack(words_logits, axis=1)
            words_predicted = tf.stack(words_predicted, axis=1)  # [max_tokens, num_sample] to [num_samples, max_tokens]
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
        output = tf.convert_to_tensor(output, dtype=tf.float64)
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
        t = time()
        states = list(zip(tf.unstack(state[0]), tf.unstack(state[1])))
        words_predicted = []
        for sample, state in zip(output, states):
            state = (tf.reshape(state[0], [1, self.num_units]), tf.reshape(state[1], [1, self.num_units]))
            sample = tf.convert_to_tensor(sample.reshape((1, 300)), dtype=tf.float64)
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
        logging.info('Beam search decoding time took = {}s'.format(round(time() - t, 3)))
        return tf.convert_to_tensor(words_predicted)

    def get_sequence(self, full_sentence):
        '''
        Slices samples in order to stop at EOS token

        Inputs:
            -> full_sentence, tensor, shape = [batch_size, max_tokens]

        Outputs:
            -> list of list, len(sublist) = sequence_length
        '''
        return [s[:s.index(self.w2i['eos'])+1] for s in full_sentence.numpy().tolist()]

    def get_loss(self, epoch, sos, state, y, sl, x, encoder_outputs, verbose=True):
        '''
        Computes loss from given batch

        Inputs:
            -> sos, numpy array, shape = [batch_size, emb_dim]
            -> state, lstm state tuple
            -> y, list of list of list

        Outputs:
            -> loss, float
        '''
        _, wl = self.forward(sos, state, x, sl, encoder_outputs, training=True)

        # +1 to sl because sl is a list of last sequence indices ie len(sequence) - 1
        sl = (np.asarray(sl) + 1).tolist()
        loss = u.cross_entropy_cost(wl, y, sequence_lengths=sl)

        if verbose and epoch != self.epoch:
            acc_words, acc_sentences = u.get_acc_word_seq(wl, y, sl)
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
