import os
import numpy as np
import tensorflow as tf
from collections import deque
import tensorflow.contrib.eager as tfe


class EncoderRNN(object):
    '''
    encoder_outputs: [max_time, batch_size, num_units] || encoder_state: [batch_size, num_units]
    '''
    def __init__(self, num_units=150, history_size=5):
        self.name = 'EncoderRNN'
        self.num_units = num_units
        self.validation_history = deque(maxlen=history_size)
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, name='encoder_lstm_cell')

    def save(self, name=None):
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        saver = tfe.Saver(self.encoder_cell.variables)
        saver.save(save_path)

    def load(self, name=None):
        self.forward(np.zeros((32, 16, 300), dtype=np.float32), list(range(2, 34, 1)))
        saver = tfe.Saver(self.encoder_cell.variables)
        name = name if name else self.name
        save_path = 'models/' + name + '/'
        saver.restore(save_path)

    def forward(self, x, sl, reverse=True):
        '''
        Performs a forward pass through a rnn

        Inputs:
            -> x, numpy array, shape = [batch_size, input_dim], example: [batch_size, sequence_length, embedding_dim]
            -> sl, list of int, list of last sequence indice for each sample in given batch
            -> reverse, boolean, optional, whether to go through the sentence in a reverse manner

        Outputs:
            -> final_output, numpy array, shape = [batch_size, cell_size]
            ps: if you want all outputs and all cell states you can acces to outputs & cell_states attribute
        '''
        state = self.encoder_cell.zero_state(len(x), dtype=tf.float32)  # Initialize LSTM cell state with zeros
        unstacked_x = tf.unstack(x, axis=1)  # unstack the embeddings, shape = [time_steps, batch_size, emb_dim]
        if reverse:
            unstacked_x = reversed(unstacked_x)
        outputs, cell_states = [], []
        for input_step in unstacked_x:
            output, state = self.encoder_cell(input_step, state)  # state = (cell_state, hidden_state = output)
            outputs.append(output)
            cell_states.append(state[0])
        # outputs shape = [time_steps, batch_size, cell_size]
        self.outputs = tf.stack(outputs, axis=1)  # Stack outputs to (batch_size, time_steps, cell_size)
        self.cell_states = tf.stack(cell_states, axis=1)

        if reverse:
            final_output = self.outputs[:,-1,:]
            final_cell_state = self.cell_states[:,-1,:]
        else:
            idxs_last_output = tf.stack([tf.range(len(x)), sl], axis=1)  # get end index of each sequence
            final_output = tf.gather_nd(self.outputs, idxs_last_output)  # retrieve last output for each sequence
            final_cell_state = tf.gather_nd(self.cell_states, idxs_last_output)
        return final_output, final_cell_state
