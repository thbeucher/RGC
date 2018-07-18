import tensorflow as tf


class DDDQN(object):
    '''
    Dueling Double Deep Q Network

    Double DQN: use mainDQN to select the action to take at the next state a'
                use targetDQN to computes the Qvalues for the next state
                -> update targetDQN, with mainDQN weights, every T steps
    '''
    def __init__(self, word2idx, idx2word, idx2emb, lstm_size=256, dense_value=512, dense_advantage=512, max_tokens=128):
        self.name = 'DDDQN'
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.max_tokens = max_tokens
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, name='lstm_cell')
        self.value_fc = tf.layers.Dense(dense_value, activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), name='value_fc')
        self.value = tf.layers.Dense(1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='value')
        self.advantage_fc = tf.layers.Dense(dense_advantage, activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='advantage_fc')
        self.advantage = tf.layers.Dense(len(word2idx), activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='advantage')
        self.to_update = [self.lstm, self.value_fc, self.value, self.advantage_fc, self.advantage]

    def update(self, update_network):
        '''
        Updates every kernel and bias of available variables with the given values

        Inputs:
            -> update_network, DDDQN instance
        '''
        for old, new in zip(self.to_update, update_network.to_update):
            old_kernel, old_bias = old.variables
            new_kernel, new_bias = new.variables
            old_kernel.assign(new_kernel)
            old_bias.assign(new_bias)

    def forward(self, input_token, lstm_state):
        '''

        Inputs:
            -> input_token, numpy array, shape = [batch_size, emb_dim]
            -> lstm_state, tuple of tensor, (cell_state, hidden_state)

        Outputs:
            -> Qvalue, tensor, shape = [batch_size]
            -> action, tensor, shape = [batch_size]
            -> lstm_state, tuple of tensor
        '''
        input_token = tf.convert_to_tensor(input_token, dtype=tf.float32)
        output, lstm_state = self.lstm(input_token, lstm_state)
        vfc = self.value_fc(output)
        v = self.value(vfc)
        afc = self.advantage_fc(output)
        a = self.advantage(afc)
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        Q = v + tf.subtract(a, tf.reduce_mean(a, axis=1, keepdims=True))
        Qvalue = tf.reduce_max(Q, axis=1)
        action = tf.argmax(Q, axis=1)
        return Qvalue, action, lstm_state


if __name__ == '__main__':
    import os
    import default
    import argparse
    import numpy as np
    import tensorflow.contrib.eager as tfe
    from data_container import DataContainer

    argparser = argparse.ArgumentParser(prog='dddqn.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    args = argparser.parse_args()

    tfe.enable_eager_execution()

    dc = DataContainer(args.input, args.emb)
    dc.prepare_data()
    mainDQN = DDDQN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens)
    targetDQN = DDDQN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens)

    x = np.asarray(dc.x_train[0])
    state = mainDQN.lstm.zero_state(dc.batch_size, dtype=tf.float32)
    for mt in range(dc.max_tokens):
        Qvalue, action, state = mainDQN.forward(x[:,mt,:], state)
        targetDQN.forward(x[:,mt,:], state)
        mainDQN.update(targetDQN)
        input('Qvalue = {} | action = {}'.format(Qvalue.numpy(), action.numpy()))
