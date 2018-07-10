import os
import ast
import sys
import logging
import argparse
import tensorflow as tf
from encoder import EncoderRNN
from decoder import DecoderRNN
import tensorflow.contrib.eager as tfe
from data_container import DataContainer
from pretrainer import parrot_initialization, see_parrot_results

import default  # to import os environment variables


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
    if rep == 'y':
        parrot_initialization(args.input, args.emb, args.attention)
