import time
import logging
import tensorflow as tf



class AttentionS2S(object):
    def __init__(self, hidden_size, type='dot', attention_size=200, attention=True):
        self.available_score_functions = {'luong': self.luong_scores, 'badhanau': self.badhanau_score, 'dot': self.dot_scores}
        self.hidden_size = hidden_size
        self.type = type
        self.attention = attention
        self.score_fn = self.available_score_functions[type]
        self.attention_layer = tf.layers.Dense(attention_size, activation=None)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = tf.convert_to_tensor(encoder_outputs, dtype=tf.float64)  # [batch_size, time_steps, cell_size]
        scores = self.score_fn(hidden, encoder_outputs)
        attention_weights = tf.nn.softmax(scores)  # [batch_size, time_steps]
        aws = tf.stack([attention_weights] * self.hidden_size, axis=2)  # [batch_size, time_steps, cell_size]
        context_vector = tf.multiply(aws, encoder_outputs)
        # expand_dims alow to go to [batch_size, 1, cell_size] from [batch_size, cell_size]
        # con = [batch_size, time_steps + 1, cell_size]
        con = tf.concat([context_vector, tf.expand_dims(hidden, axis=1)], 1)
        attention_vector = self.attention_layer(con) if self.attention else con
        return attention_vector

    def luong_scores(self, hidden, encoder_outputs):
        attention = tf.layers.dense(hidden, self.hidden_size, activation=None, use_bias=False)
        scores = tf.einsum('ij,klj', hidden, encoder_outputs)
        idx_to_keep = [[i, i] for i in range(scores.shape[0])]
        return tf.gather_nd(scores, idx_to_keep)

    def dot_scores(self, hidden, encoder_outputs):
        '''
        hidden -> [batch_size, cell_size]
        encoder_outputs -> [batch_size, time_steps, cell_size]

        return -> [batch_size, time_steps]
        it returns one weight for each time step for each given samples
        '''
        scores = tf.einsum('ij,klj', hidden, encoder_outputs)
        idx_to_keep = [[i, i] for i in range(scores.shape[0])]
        return tf.gather_nd(scores, idx_to_keep)

    def badhanau_score(self, hidden, encoder_output):
        pass
