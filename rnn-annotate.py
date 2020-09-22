import torch
import argparse

from classes import tagger as tg

parser = argparse.ArgumentParser()
parser.add_argument('path_param', type=str, help='Set path to initialize the data object and the stored RNN tagger model.')
parser.add_argument('path_sents', type=str, help='Set path to sentences to annontate')
args = parser.parse_args()


def annotate(path, data, tagger):
    sents_generator = data.sentences(path)
    for sent in sents_generator:
        output = tagger(torch.LongTensor(data.words2IDs(sent)))
    yield data.IDs2tags(output)
        
if __name__ == '__main__':
    
    data = tg.Data(args.path_param+'.io')
    model = torch.load(args.path_param+'.rnn')
    output_generator = annotate(args.path_sents, data, model)