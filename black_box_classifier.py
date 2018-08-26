import os
import regex
import numpy as np
from sklearn.svm import LinearSVC
from data_container import DataContainer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

import default


class BlackBoxClassifier(object):
  def __init__(self, language='en', test_size=0.2, dc=None, input_file=None, prepare_data=False):
    self.test_size = test_size
    if dc is None:
      self.dc = DataContainer(input_file, '')
    else:
      self.dc = dc
    self.vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, smooth_idf=True, tokenizer=lambda x: x.split(' '))
    self.classifier = LinearSVC(tol=0.5)
    if prepare_data:
      self.prepare_data(self.dc.sources, self.dc.labels)

  def clean_sentence(self, sentence):
    '''
    Removes punctuations and double space
    '''
    return regex.sub(r' +', ' ', regex.sub(r'\p{Punct}', '', sentence)).strip()

  def prepare_data(self, sources, labels, clean=True):
    '''
    Cleans sentences then split it into train/test

    Inputs:
      -> sources, list of string
      -> labels, list of string

    Data enabled:
      -> x_train, list of string
      -> x_test, list of string
      -> y_train, list of string
      -> y_test, list of string
    '''
    if clean:
      sources = [self.clean_sentence(s) for s in sources]
    self.x_train, self.x_test, self.y_train, self.y_test =\
    train_test_split(sources, labels, test_size=self.test_size, stratify=labels)

  def train(self, x_train, y_train):
    '''
    Trains the classifier

    Inputs:
      -> x_train, list of string
      -> y_train, list of string
    '''
    train = self.vectorizer.fit_transform(x_train)
    self.classifier.fit(train, y_train)

  def predict(self, x_test):
    '''
    Gets predictions from the classifier for the given inputs

    Inputs:
      -> x_test, list of string

    Outputs:
      -> list of string, predicted labels for the given sentences
    '''
    test = self.vectorizer.transform(x_test)
    return self.classifier.predict(test)

  def predict_test(self, x_test, y_test):
    test = self.vectorizer.transform(x_test)
    preds = self.classifier.predict(test)
    print(classification_report(preds, y_test))
    mispredicted = np.where(preds != y_test)[0]
    return mispredicted, f1_score(preds, y_test, average='weighted')

  def get_accuracy_f1(self, x_test, y_test):
    test = self.vectorizer.transform(x_test)
    preds = self.classifier.predict(test)
    acc = accuracy_score(preds, y_test)
    f1 = f1_score(preds, y_test, average='weighted')
    return acc, f1

  def get_reward(self, sentence, label, terminal=False):
    sv = self.vectorizer.transform([sentence])
    pred = self.classifier.predict(sv)[0]
    num_tokens = len(sentence.split(' '))
    if terminal:
      return 100 + 10 / num_tokens if pred == label else -10 - num_tokens
    else:
      return 10 if pred == label else -1


if __name__ == '__main__':
  import argparse
  argparser = argparse.ArgumentParser(prog='black_box_classifier.py', description='')
  argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
  args = argparser.parse_args()

  b = BlackBoxClassifier(input_file=args.input, prepare_data=True, test_size=0.2)
  b.train(b.x_train, b.y_train)
  b.predict_test(b.x_test, b.y_test)
