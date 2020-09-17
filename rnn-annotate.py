import torch

from classes import tagger as tg

data = tg.Data(args.path_param+'.io')
model = torch.load(args.path_param+'.rnn')