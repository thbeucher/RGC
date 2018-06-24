import tqdm
import regex
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
