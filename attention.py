import time
import logging
import tensorflow as tf



class AttentionS2S(object):
    def __init__(self, hidden_size, type='luong'):
        self.available_score_functions = {'luong': self.luong_score, 'badhanau': self.badhanau_score}
        self.hidden_size = hidden_size
        self.type = type
        self.score_fn = self.available_score_functions[type]

    def forward(self, hidden, encoder_outputs):
        '''
        '''
        # hidden shape = [batch_size, cell_size]
        # encoder_outputs shape = [batch_size, time_steps, cell_size]
        encoder_outputs = tf.unstack(encoder_outputs, axis=1)
        scores = [self.score_fn(hidden, eh) for eh in encoder_outputs]  # very slow
        attention_weights = tf.nn.softmax(scores)
        # attention_weights shape = [time_steps, batch_size, cell_size]
        context_vector = tf.multiply(attention_weights, encoder_outputs)
        # context_vector shape = [time_steps, batch_size, cell_size]
        context_vector = tf.unstack(context_vector, axis=0)
        context_vector.append(hidden)
        attention_vector = tf.stack(context_vector, axis=1)
        # attention_vector shape = [batch_size, time_steps + 1, cell_size]
        return attention_vector

    def luong_score(self, hidden, encoder_output):
        attention = tf.layers.dense(hidden, self.hidden_size, activation=None, use_bias=False)
        # attention shape = [batch_size, cell_size]
        return tf.multiply(attention, encoder_output)

    def badhanau_score(self, hidden, encoder_output):
        pass
