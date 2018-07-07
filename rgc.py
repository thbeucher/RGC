import os
import ast
import sys
import logging
import argparse
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from pretrainer import parrot_initialization

import default  # to import os environment variables


if __name__ == '__main__':
    # # TODO:
    # => Add save & load of model
    #   -> currently, encoder save & load works but not Encoder-Decoder
    #
    # => Luong attention implemented but slow and maybe not good
    argparser = argparse.ArgumentParser(prog='rgc.py', description='')
    argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
    argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
    argparser.add_argument('--key', metavar='KEY', default='test', type=str)
    argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
    argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tfe.enable_eager_execution()

    parrot_initialization(args.input, args.emb, args.attention)
