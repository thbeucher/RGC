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
        self.get_data_from_json(input_file)

    def get_data_from_json(self, input_file):
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

    def prepare_data(self):
        '''
        Loads & prepares training data into batches
        '''
        logging.info('Loads embeddings...')
        self.emb = ft.load_model(self.emb_path)
        logging.info('Embeddings loaded.')
        cleaned_sources = [u.clean_sentence(s) + ' eos' for s in self.sources]  # add EndOfSentence token
        decoded_lowered_sources = [unidecode(s).lower() for s in cleaned_sources]  # decode and lowerize sentences
        self.sos = np.vstack([self.emb['sos']] * self.batch_size)  # get embedding of SOS token and prepare it to batch size
        sources = u.to_emb(decoded_lowered_sources, self.emb)  # list of numpy array [num_tokens, embeddings_size]
        sources, seq_lengths, max_sl = u.pad_data(sources, pad_with=self.emb['eos'])
        labels, self.num_class = u.encode_labels(self.labels)
        self.max_tokens = max_sl

        # add StartOfSentence token to vocabulary
        rep = input('Load dictionaries? (y or n): ')
        if rep == 'y':
            self.load_dicts()
        else:
            self.vocabulary, self.word2idx, self.idx2word, self.idx2emb = u.get_vocabulary(decoded_lowered_sources + ['sos'], self.emb)
            self.word2onehot = u.one_hot_encoding(list(self.word2idx.keys()))
            self.save_dicts()
        self.y_parrot, self.y_parrot_padded = self.get_y_parrot(decoded_lowered_sources)

        # _tr = _train | _te = _test
        x_tr, self.x_te, y_tr, self.y_te, sl_tr, self.sl_te, y_p_tr, self.y_p_te, y_p_p_tr,\
        self.y_p_p_te = train_test_split(sources, labels, seq_lengths, self.y_parrot, self.y_parrot_padded,
                                         test_size=self.test_size, stratify=self.labels)

        sources, labels, seq_lengths, y_parrot_shuffled, y_p_p_s = u.shuffle_data(x_tr, y_tr, sl_tr, y_p_tr, y_p_p_tr)
        self.y_parrot_batch = u.create_batch(y_parrot_shuffled, batch_size=self.batch_size)
        self.y_parrot_padded_batch = u.create_batch(y_p_p_s, batch_size=self.batch_size)
        self.x_train, self.y_train, self.sl_train = u.to_batch(sources, labels, seq_lengths, batch_size=self.batch_size)

    def get_y_parrot(self, sources, pad_with='eos'):
        '''
        Gets y for parrot initilization

        Inputs:
            -> sources, list of string
            -> pad_with, string, optional, default to `eos`

        Outputs:
            -> y_parrot, list of list, sublists are list of one hot vector
            -> y_parrot_padded, list of list, same as y_parrot but each sublist are filled with
               supplementary one hot vector (one hot vector of given pad_with value) in order to
               have each sublist of the same len
        '''
        y_parrot, y_parrot_padded = [], []
        for s in sources:
            tmp = [self.word2onehot[w] for w in s.split(' ')]
            y_parrot.append(tmp)

            tmp_pad = tmp + [self.word2onehot[pad_with]] * (self.max_tokens - len(tmp))
            y_parrot_padded.append(tmp_pad)
        return y_parrot, y_parrot_padded

    def get_sos_batch_size(self, batch_size):
        return np.vstack([self.emb['sos']] * batch_size)

    def shuffle_training(self):
        self.x_train, self.y_train, self.sl_train = u.shuffle_data(self.x_train, self.y_train, self.sl_train)
