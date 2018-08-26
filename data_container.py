import json
import logging
import numpy as np
import utility as u
import pickle as pk
import fasttext as ft
from unidecode import unidecode
from sklearn.model_selection import train_test_split


class DataContainer(object):
    def __init__(self, input_file, emb_path, batch_size=32, test_size=0.2):
        '''
        Inputs:
            -> input_file, string
            -> emb_path, string
            -> batch_size, int, optional, default value to 32
            -> test_size, float between 0 and 1, optional, proportion of the dataset to include to test split
        '''
        self.input_file = input_file
        self.emb_path = emb_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.get_data_from_json(input_file)

    def get_data_from_json(self, input_file):
        '''
        Reads json dataset and extract sources & labels

        Inputs:
            -> input_file, string, path to the json file

        After called the function you can acces to the sources & labels variables
        through the class instance
        '''
        with open(input_file, 'r') as f:
            self.dataset = json.load(f)
        self.sources, self.labels = [], []
        for intent in self.dataset['intents']:
            for exp in intent['expressions']:
                self.sources.append(exp['source'])
                self.labels.append(intent['name'])

    def save_dicts(self, path='data/'):
        with open(path + 'vocab.pk', 'wb') as f:
            pk.dump(self.vocabulary, f)
        with open(path + 'w2i.pk', 'wb') as f:
            pk.dump(self.word2idx, f)
        with open(path + 'i2w.pk', 'wb') as f:
            pk.dump(self.idx2word, f)
        with open(path + 'i2e.pk', 'wb') as f:
            pk.dump(self.idx2emb, f)
        with open(path + 'w2o.pk', 'wb') as f:
            pk.dump(self.word2onehot, f)

    def load_dicts(self, path='data/'):
        with open(path + 'vocab.pk', 'rb') as f:
            self.vocabulary = pk.load(f)
        with open(path + 'w2i.pk', 'rb') as f:
            self.word2idx = pk.load(f)
        with open(path + 'i2w.pk', 'rb') as f:
            self.idx2word = pk.load(f)
        with open(path + 'i2e.pk', 'rb') as f:
            self.idx2emb = pk.load(f)
        with open(path + 'w2o.pk', 'rb') as f:
            self.word2onehot = pk.load(f)

    def get_load_save_vocab_data(self):
        '''
        Loads or creates & saved data about vocabulary

        Available data after calling this function:
            -> vocabulary, list of string, list of words
            -> word2idx, dictionary,
            -> idx2word, dictionary,
            -> idx2emb, dictionary,
            -> word2onehot, dictionary,
        '''
        rep = input('Load dictionaries? (y or n): ')
        if rep == 'y' or rep == '':
            self.load_dicts()
        else:
            self.vocabulary, self.word2idx, self.idx2word, self.idx2emb = u.get_vocabulary(decoded_lowered_sources + ['sos'], self.emb)
            self.word2onehot = u.one_hot_encoding(list(self.word2idx.keys()))
            self.save_dicts()

    def preprocess_sentences(self, sentences):
        '''
        Remove double space and punctutation then add EOS token and finally
        remove accent then lowerize

        Inputs:
            -> sentences, list of string, list of sentences to preprocess

        Outputs:
            -> cleaned_source, list of string
            -> decoded_lowered_sources, list of string
        '''
        cleaned_sources = [u.clean_sentence(s) + ' eos' for s in sentences]
        decoded_lowered_sources = [unidecode(s).lower() for s in cleaned_sources]
        return cleaned_sources, decoded_lowered_sources

    def source_to_emb(self, sources, pad_with='eos', emb=None):
        '''
        Converts sources to embedding representation, option to pad with one specific token

        Inputs:
            -> sources, list of string
            -> pad_with, optional, string, token to use to pad sources

        Outputs:
            -> padded_emb_sources, list of numpy array
            -> seq_lengths, list of int
            -> max_sl, int, the bigger sequence length
        '''
        if emb is None:
          logging.info('Loads embeddings...')
          self.emb = ft.load_model(self.emb_path)
          logging.info('Embeddings loaded.')
        else:
          self.emb = emb
        emb_sources = u.to_emb(sources, self.emb)  # list of numpy array [num_tokens, embeddings_size]
        padded_emb_sources, seq_lengths, max_sl = u.pad_data(emb_sources, pad_with=self.emb[pad_with])
        return padded_emb_sources, seq_lengths, max_sl

    def get_y_classif(self, labels):
        '''
        Returns expected output for classification, one hot encoding of each sample outputs

        Inputs:
            -> labels, list of string

        Outputs:
            -> y, numpy array, shape = [num_samples, num_labels]
            -> num_class, int
        '''
        return u.encode_labels(labels)

    def get_y_sources(self, sources, pad_with=None, max_tokens=None):
        '''
        Gets expected output as sequence ie y will reflect the tokens to find for each sources

        Inputs:
            -> sources, list of string
            -> pad_with, string, token to use to pad the sequences
            -> max_tokens, int, the maximum number of tokens for each sequences

        Outputs:
            -> y_parrot, list of list of list, [[one_hot_vector], [...], ...]
               shape = [num_samples, num_tokens, vocabulary_size]
            OR
            -> y_parrot_padded, same as y_parrot but with sequence length fixed
               shape = [num_samples, max_tokens, vocabulary_size]
        '''
        y_parrot = [[self.word2onehot[w] for w in s.split(' ')] for s in sources]
        if pad_with is not None:
            y_parrot_padded = [yp + [self.word2onehot[pad_with]] * (max_tokens - len(yp)) for yp in y_parrot]
            return y_parrot_padded
        return y_parrot

    def split_data(self, *args, stratify_ref=None):
        '''
        Splits given data into train/test

        Inputs:
            -> args, give as many list as you want
            -> stratify_ref, list, labels to use in order to homogenously split the data

        Outputs:
            -> a list of train/test data, len(list) = 2*len(args)
        '''
        stratify = self.labels if stratify_ref is None else stratify_ref
        return train_test_split(*args, test_size=self.test_size, stratify=stratify)

    def transform_sources(self, sources, emb=None):
      '''

      Inputs:
        -> sources, list of string,

      Outputs:
        -> sources, list of numpy array
        -> seq_lengths, list of int
        -> decoded_lowered_sources, list of string
        -> max_tokens, int
      '''
      _, decoded_lowered_sources = self.preprocess_sentences(sources)
      sources, seq_lengths, max_tokens = self.source_to_emb(decoded_lowered_sources, emb=emb)
      return sources, seq_lengths, decoded_lowered_sources, max_tokens

    def prepare_data(self):
        '''
        Loads & prepares all data
        '''
        sources, seq_lengths, decoded_lowered_sources, self.max_tokens = self.transform_sources(self.sources)
        y_classif, self.num_class = self.get_y_classif(self.labels)

        self.get_load_save_vocab_data()
        y_parrot = self.get_y_sources(decoded_lowered_sources)
        y_parrot_padded = self.get_y_sources(decoded_lowered_sources, pad_with='eos', max_tokens=self.max_tokens)

        # shuffle all data | dls = decoded_lowered_sources
        self.x, self.y_classif, self.sl, self.y_parrot, self.y_parrot_padded, self.dls, self.labels\
          = u.shuffle_data(sources, y_classif, seq_lengths, y_parrot, y_parrot_padded, decoded_lowered_sources, self.labels)
        # split into train test
        # _tr = _train | _te = _test
        self.x_tr, self.x_te, self.y_tr_classif, self.y_te_classif, self.sl_tr, self.sl_te, self.y_p_p_tr, self.y_p_p_te,\
          self.dls_tr, self.dls_te = self.split_data(self.x, self.y_classif, self.sl, self.y_parrot_padded, self.dls)

        # arrange training data into batchs
        self.x_train, self.y_train_classif, self.sl_train, self.y_parrot_padded_train = u.to_batch(self.x_tr, self.y_tr_classif,
                                                                                                   self.sl_tr, self.y_p_p_tr,
                                                                                                   batch_size=self.batch_size)

    def get_sos_batch_size(self, batch_size):
        return np.vstack([self.emb['sos']] * batch_size)
