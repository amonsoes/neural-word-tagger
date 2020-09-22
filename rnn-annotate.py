import torch
import argparse

from classes import tagger as tg

parser = argparse.ArgumentParser()
parser.add_argument('path_param', type=str, help='Set path to initialize the data object and the stored RNN tagger model.')
args = parser.parse_args()

data = tg.Data(args.path_param+'.io')
model = torch.load(args.path_param+'.rnn')