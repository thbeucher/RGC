import os
import ast
import sys
import logging
import argparse
import tensorflow as tf
from dddqn import DDDQN
from encoder import EncoderRNN
import tensorflow.contrib.eager as tfe
from data_container import DataContainer
from pretrainer import parrot_initialization_rgc

import default  # to import os environment variables


def load_needed(dataset, emb_path, attention):
    dc = DataContainer(dataset, emb_path)
    dc.prepare_data()
    encoder = EncoderRNN()
    dddqn = DDDQN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)
    encoder.load(name='Encoder-Decoder/Encoder')
    dddqn.load(name='Encoder-Decoder/Decoder')
    return encoder, dddqn, dc


if __name__ == '__main__':
    # # TODO:
    # => Luong attention implemented but the attention vector is not feed to the decoder input
    argparser = argparse.ArgumentParser(prog='rgc.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
    argparser.add_argument('--key', metavar='KEY', default='test', type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    rep = input('Launch parrot initialization? (y or n): ')
    if rep == 'y' or rep == '':
        logging.info('Launch of parrot initilization...')
        encoder, dddqn, dc = parrot_initialization_rgc(args.input, args.emb)
    else:
        logging.info('Load previously pretrained encoder-decoder...')
        encoder, decoder, dc = load_needed(args.input, args.emb)
