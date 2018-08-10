import os
import ast
import sys
import logging
import argparse
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from rgc import RGC


import default


def main_execution(dataset, emb_path):
  DQNetwork = RGC(dataset, emb_path, name='DQNetwork')
  TargetNetwork = RGC(dataset, emb_path, dc=DQNetwork.dc, bbc=DQNetwork.bbc, name='TargetNetwork')

  DQNetwork.pretrain()
  DQNetwork.test_pretrained()

  TargetNetwork.update(DQNetwork, init_layers=True)
  TargetNetwork.test_pretrained()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='run_rgc.py', description='')
  argparser.add_argument('--input', metavar='INPUT', default=os.environ['INPUT'], type=str)
  argparser.add_argument('--language', metavar='LANGUAGE', default='en', type=str)
  argparser.add_argument('--key', metavar='KEY', default='test', type=str)
  argparser.add_argument('--emb', metavar='EMB', default=os.environ['EMB'], type=str)
  argparser.add_argument('--attention', metavar='ATTENTION', default='False', type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  tfe.enable_eager_execution()

  main_execution(args.input, args.emb)
