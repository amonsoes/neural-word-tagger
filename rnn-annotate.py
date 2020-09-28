import torch
import argparse

from classes import tagger as tg
from classes import tools

def annotate(path, data, tagger):
    sents = []
    sents_generator = data.sentences(path)
    for sent in sents_generator:
        output = tagger(torch.LongTensor(data.words2IDs(sent)))
        sents.append((sent,output))
    return sents
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_param', type=str, help='Set path to initialize the data object and the stored RNN tagger model.')
    parser.add_argument('--path_sents', type=str, help='Set path to sentences to annontate')
    parser.add_argument('--gpu', type= tools.str2bool, help='set if you have cuda available')

    args = parser.parse_args()
    
    data = tg.Data(args.path_param+'.io')
    model = torch.load(args.path_param+'.rnn') if args.gpu else torch.load(args.path_param+'.rnn', map_location=torch.device('cpu') )
    if not args.gpu:
        model.device = torch.device('cpu')
    output_return = annotate(args.path_sents, data, model)
    for sent, output in output_return:
        tags = data.IDs2tags([int(torch.argmax(tensor)) for tensor in output])
        for i in range(0,len(sent)):
            print(sent[i]," : ", tags[i])