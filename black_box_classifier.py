import os
import regex
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import default


class BlackBoxClassifier(object):
    def __init__(self, input_file, language='en', test_size=0.2):
        self.test_size = test_size
        self.dc = DataContainer(input_file, '')
        with open(os.environ['STOPWORDS'].format(language), 'r') as f:
            self.stopwords = set(f.read().splitlines())
        self.vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, smooth_idf=True,
                                          stop_words=self.stopwords, tokenizer=lambda x: x.split(' '))
        self.classifier = LinearSVC(tol=0.5)
        self.prepare_data(self.dc.sources, self.dc.labels)

    def clean_sentence(self, sentence):
        return regex.sub(r' +', ' ', regex.sub(r'\p{Punct}', '', sentence)).strip()

    def prepare_data(self, sources, labels):
        cleaned_sources = [self.clean_sentence(s) for s in sources]
        self.x_train, self.x_test, self.y_train, self.y_test =\
        train_test_split(cleaned_sources, labels, test_size=self.test_size, stratify=labels)

    def train(self, x_train, y_train):
        train = self.vectorizer.fit_transform(x_train)
        self.classifier.fit(train, y_train)

    def predict_test(self, x_test, y_test):
        test = self.vectorizer.transform(x_test)
        preds = self.classifier.predict(test)
        print(classification_report(preds, y_test))

    def get_reward(self, x_test, y_test):
        test = self.vectorizer.transform(x_test)
        preds = self.classifier.predict(test)
        return sum(preds == y_test)


if __name__ == '__main__':
    b = BlackBoxClassifier(os.environ['INPUT'])
    b.train(b.x_train, b.y_train)
    b.predict_test(b.x_test, b.y_test)
    print(b.get_reward(b.x_test, b.y_test))
