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
from pretrainer import parrot_initialization_encoder_decoder, see_parrot_results

import default  # to import os environment variables


def load_needed(dataset, emb_path, attention):
    dc = DataContainer(dataset, emb_path)
    dc.prepare_data()
    encoder = EncoderRNN()
    decoder = DecoderRNN(dc.word2idx, dc.idx2word, dc.idx2emb, max_tokens=dc.max_tokens, attention=attention)
    encoder.load(name='Encoder-Decoder/Encoder')
    decoder.load(name='Encoder-Decoder/Decoder')
    return encoder, decoder, dc


if __name__ == '__main__':
    # # TODO:
    # => Luong attention implemented but the attention vector is not feed to the decoder input
    #
    # => add connection between encoder-decoder & bbc for loss computation
    # => add network to estimate reward at each decoding step
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
        encoder, decoder, dc = parrot_initialization_encoder_decoder(args.input, args.emb, args.attention)
    else:
        logging.info('Load previously pretrained encoder-decoder...')
        encoder, decoder, dc = load_needed(args.input, args.emb, args.attention)
