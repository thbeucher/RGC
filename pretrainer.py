import logging
import numpy as np
import tensorflow as tf
from collections import deque


class PreTrainer(object):
    def __init__(self, num_class, layer_to_train, history_size=5):
        self.num_class = num_class
        self.layer_to_train = layer_to_train
        self.early_stoping = False
        self.epoch = -1
        self.validation_history = deque(maxlen=history_size)
        self.optimizer = tf.train.AdamOptimizer()
        self.prediction_layer = tf.layers.Dense(num_class, activation=None, name='prediction_layer')

    def forward(self, x):
        '''
        Performs a forward pass then go through a prediction layer to get predicted classes

        Inputs:
            -> x, numpy array, shape = [batch_size, input_dim]

        Outputs:
            -> logits, tensor, shape = [batch_size, num_class]
        '''
        state = self.layer_to_train.zero_state(len(x), dtype=tf.float32)  # Initialize LSTM cell state with zeros
        unstacked_x = tf.unstack(x, axis=1)  # unstack the embeddings, shape = [time_steps, batch_size, emb_dim]
        unstacked_x = reversed(unstacked_x)
        outputs = []
        for input_step in unstacked_x:
            output, state = self.layer_to_train(input_step, state)  # state = (cell_state, hidden_state = output)
            outputs.append(output)
        outputs = tf.stack(outputs, axis=1)  # reshape from [times, batch_size, num_units] to [batch_size, times, num_units]
        logits = self.prediction_layer(output)
        return logits

    def get_loss(self, epoch, x, y, verbose=True):
        '''
        Computes the loss from a forward pass

        Inputs:
            -> epoch, int
            -> x, numpy array, shape = [batch_size, input_dim]
            -> y, numpy array, shape = [batch_size, num_class]
            -> verbose, boolean, decide wether or not print the loss & accuracy at each epoch
        '''
        y_loss = [el.tolist().index(1) for el in y]  # if using sparse_softmax_cross_entropy_with_logits
        logits = self.forward(x)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_loss, logits=logits)
        loss = tf.reduce_mean(loss)

        if verbose and epoch != self.epoch:
            predict = tf.argmax(logits, 1).numpy()
            target = np.argmax(y, 1)
            accuracy = np.sum(predict == target) / len(target)
            logging.info('Epoch {} -> loss = {} | acc = {}'.format(epoch, loss, round(accuracy, 3)))
            self.epoch += 1

        return loss

    def classification_train(self, x, y):
        '''
            x = [x_train, x_test]
            y = [y_train, y_test]
        '''
        x_train, x_test = x
        y_train, y_test = y
        for epoch in range(300):
            for x, y in zip(x_train, y_train):
                self.optimizer.minimize(lambda: self.get_loss(epoch, x, y))
            acc = self.validation(epoch, x_test, y_test)
            if self.early_stoping:
                break
        return acc

    def validation(self, epoch, x_test, y_test):
        '''
        Computes validation accuracy and define an early stoping

        Inputs:
            -> epoch, int
            -> x_test, numpy array, shape = [num_test_samples, input_dim]
            -> y_test, numpy array, shape = [num_test_samples, num_class]

        Outputs:
            Print validation accuracy and set a boolean (early_stoping) to True
            if stoping criterias are filled

            Stoping criterias:
                -> new validation accuracy is lower than all accuracies stored in validation_history
                -> new accuracy is equal to all accuracies stored in validation_history
        '''
        logits = self.forward(x_test)
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
