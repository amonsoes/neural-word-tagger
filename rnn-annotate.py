import torch
import argparse

from classes import tagger as tg


def annotate(path, data, tagger):
    sents_generator = data.sentences(path)
    for sent in sents_generator:
        output = tagger(torch.LongTensor(data.words2IDs(sent)))
        yield (sent,data.IDs2tags(output))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_param', type=str, help='Set path to initialize the data object and the stored RNN tagger model.')
    parser.add_argument('--path_sents', type=str, help='Set path to sentences to annontate')
    parser.add_argument('--gpu', type= tg.str2bool, help='set if you have cuda available')

    args = parser.parse_args()
    
    data = tg.Data(args.path_param+'.io')
    model = torch.load(args.path_param+'.rnn') if args.gpu else torch.load(args.path_param+'.rnn', map_location=torch.device('cpu') )
    output_generator = annotate(args.path_sents, data, model)
    for sent, output in output_generator:
        for i in range(0,len(sent)):
            print(sent[i],output[i])