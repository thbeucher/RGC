import os
import heapq
import logging
import numpy as np
from time import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe


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

    def load(self, name=None, only_lstm=False):
        sos = np.zeros((32, 300), dtype=np.float32)
        state = self.decoder_cell.zero_state(32, dtype=tf.float32)
        self.forward(sos, state, [], list(range(2, 34, 1)))
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
                words_predicted = self.beam_search(sos, state)  # beam_search decoding with beam_size of 1 = greedy_decoding
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
        t = time()
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
        logging.info('Beam search decoding time took = {}s'.format(round(time() - t, 3)))
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

        # +1 to sl because sl is a list of last sequence indices ie len(sequence) - 1
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
