import tqdm
import regex
import random
import logging
import numpy as np
import tensorflow as tf
from unidecode import unidecode


def create_batch(iterable, batch_size=None, num_batch=None):
    '''
    Creates list of list
        if you provide batch_size, sublist will be of size batch_size
        if you provide num_batch, it will split your data into num_batch lists
    '''
    size = len(iterable)
    if num_batch:
        assert size >= num_batch, 'There will have empty batch if len(iterable) < num_batch'
        return [iterable[i::num_batch] for i in range(num_batch)]
    else:
        return [iterable[i:i+batch_size] for i in range(0, size, batch_size)]


def to_batch(*args, batch_size=1):
    '''
    Transforms data into batchs

    Inputs:
        -> give as many list as you want
        -> batch_size, int, optional

    Outputs:
        -> your list transformed into batches
    '''
    logging.info('Transform data into batch...')
    return [create_batch(l, batch_size=batch_size) for l in args]


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


def pad_data(sources, pad_with=None):
    '''
    Pads sequences to the same length defined by the biggest one

    Inputs:
        -> sources, list of numpy array of shape [sequence_length, embedding_dim]
        -> pad_with, optional, if given, must be a list or numpy array of shape = [emb_dim]
           leave to None to pad with zeros

    Outputs:
        -> sentences, list of numpy array, padded sequences
        -> sequence_lengths, list of int, list of last sequence indice for each sequences
        -> max_seq_length, size of the longer sentence
    '''
    logging.info('Padding sources...')
    sequence_lengths = [s.shape[0] - 1 for s in sources]
    max_seq_length = np.max(sequence_lengths) + 1
    if pad_with is not None:
        sentences = [np.vstack([s] + [pad_with] * (max_seq_length - len(s))) for s in sources]
    else:
        sentences = [np.pad(s, pad_width=[(0, max_seq_length - s.shape[0]), (0, 0)], mode='constant', constant_values=0) for s in sources]
    return sentences, sequence_lengths, max_seq_length


def shuffle_data(*data):
    '''
    Shuffles data

    Inputs:
        -> give as many list as you want, all list must be of the same size

    Outputs:
        -> list of shuffled lists
    '''
    logging.info('Shuffling data...')
    to_shuffle = list(zip(*data))
    random.shuffle(to_shuffle)
    return [list(map(lambda x: x[i], to_shuffle)) for i in range(len(data))]


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


def one_hot_encoding(labels):
    '''
    Maps one hot encodings of labels with labels name

    Inputs:
        -> labels, list of string, list of labels

    Outputs:
        -> labels2onehot, dictionary, map between label name and his one hot encoding
    '''
    uniq_labels = list(set(labels))
    labels2onehot = {}
    for l in uniq_labels:
        onehot = np.zeros(len(uniq_labels), dtype=np.float32)
        onehot[uniq_labels.index(l)] = 1.
        labels2onehot[l] = onehot
    return labels2onehot


def get_vocabulary(sources, emb, unidecode_lower=False):
    '''
    Gets vocabulary from the sources & creates word to index and index to word dictionaries

    Inputs:
        -> sources, list of string (to be interesting, you must clean your sentences before)
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


def cross_entropy_cost(output, target, sequence_lengths=None):
    '''
    Computes the cross entropy loss with respect to the given sequence lengths.

    Cross entropy indicates the distance between what the model believes the output distribution should be,
    and what the original distribution really is.

    Suppose you have a fixed model which predict for n class their hypothetical occurence
    probabilities y1, y2, ..., yn. Suppose that you now observe (in reality) k1 instance of
    class1, k2 instance of class2, ..., kn instance of classn. According to your model,
    the log likelihood of this happening is P(data|model) = y1^k1.y2^k2...yn^kn
    Take the log and changind the sign: -log(P(data|model)) = -sum( ki.log(yi) )i (ie sum on i)
    If you now divide the righ-hand sum by the number of observations N = k1 + k2 + ... + kn
    and denote the empirical probabilities as ÿi = ki/N, you'll get the cross-entropy:
    -(1/N)log(P(data|model)) = -(1/N)sum( ki.log(yi) )i = - sum( ÿi.log(y) )i

    Inputs:
        -> output, tensor, the word logits, shape = [batch_size, num_tokens, vocab_size]
        -> target, tensor or numpy array or list, same shape as output
        -> sequence_length, list of the sequence lengths

    Outputs:
        -> float, the cross entropy loss
    '''
    # if we do not clip the value, it produces NAN
    cross_entropy = target * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)

    if sequence_lengths is not None:
        mask = tf.cast(tf.sequence_mask(sequence_lengths, output.shape[1]), dtype=tf.float32)
        # mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
        cross_entropy *= mask

    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)


def init_rnn_layer(layer_to_init):
    '''
    Initialize an rnn layer

    Inputs:
        -> layer_to_init, tensorflow rnn layer
    '''
    hidden = tf.convert_to_tensor(np.zeros((32, 300), dtype=np.float32))
    state = layer_to_init.zero_state(32, dtype=tf.float32)
    layer_to_init(hidden, state)


def update_layer(layer_to_update, new_values):
    '''
    Updates the given layer trainables with the given values

    Inputs:
        -> layer_to_update, tensorflow layer
        -> new_values, tensorflow layer
    '''
    old_kernel, old_bias = layer_to_update.variables
    new_kernel, new_bias = new_values.variables
    old_kernel.assign(new_kernel)
    old_bias.assign(new_bias)
