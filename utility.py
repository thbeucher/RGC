import tqdm
import regex
import random
import logging
import numpy as np


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
