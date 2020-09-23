import random
import torch
import argparse

from sys import argv
from classes import tagger as tg

def train(data, tagger, numEpochs):
    if args.gpu:
        tagger.cuda()
    else:
        print('\nWARNING: Cuda not available. Training initialized on CPU\n')
    optimizier = torch.optim.Adam(tagger.parameters(), lr=args.learning_rate)
    best_current_acc = 0.0
    for epoch in range(numEpochs):
        for x, y in data.trainSentences:
            output = tagger(torch.LongTensor(data.words2IDs(x)))
            loss = torch.nn.CrossEntropyLoss()
            loss_output = loss(output, torch.LongTensor(data.tags2IDs(y)))
            loss_output.backward()
            optimizier.step()
        random.shuffle(data.trainSentences)
        tagger.train(mode=False)
        total_tagged, sum_corr = 0, 0
        for x, y in data.devSentences:
            total_tagged += len(y)
            output = tagger(torch.LongTensor(data.words2IDs(x)))
            sum_corr += sum([1 for ix,iy in zip(output,torch.LongTensor(data.tags2IDs(y))) if torch.argmax(ix).item() == iy.item()])
        accuracy = sum_corr / total_tagged
        if accuracy > best_current_acc:
            best_current_acc = accuracy
            torch.save(tagger, args.parfile+'.rnn')

if __name__ == '__main__':
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='Set hyperparams for tagger RNN')

    #positional args
    parser.add_argument('trainfile', type=str , help='set training data')
    parser.add_argument('devfile', type=str, help='set development data')
    parser.add_argument('parfile', type=str, help='set file to load/dump data attributes. Needs suffix')

    #optional args
    parser.add_argument('--num_epochs', type=int, help='set the number of epochs of the training')
    parser.add_argument('--num_words', type=int, help='set the number of words. This will impact the training speed, but also affects quality')
    parser.add_argument('--emb_size', type=int, help='set the number of dimensions of the embedding matrix. This will impact the training speed, but also affects quality')
    parser.add_argument('--rnn_size', type=int, help='set the number of dimensions of the LSTM vector. This will impact the training speed, but also affects quality')
    parser.add_argument('--dropout_rate', type=float, help='set the dropout rate')
    parser.add_argument('--learning_rate', type=float, help='set the learning rate of the optimizer')
    parser.add_argument('--gpu', type=str2bool, nargs='?', const=True, default=False, help='set True if cuda-able GPU is available. Else set False')

    args = parser.parse_args()
    
    print('Initisalizing training...\n\n Parameters:\n')
    print('  NUMWORDS : {}\n  EMBSIZE : {}\n  RNNSIZE : {}\n  NUMEPOCHS :{}\n  DO_RATE :{}\n  L_RATE : {}\n  CUDA : {}\n\n'.format(args.num_words, args.emb_size, args.rnn_size, args.num_epochs, args.dropout_rate, args.learning_rate, args.gpu))

    data = tg.Data(args.trainfile, args.devfile, args.num_words)
    tagger = tg.TaggerModel(args.num_words, data.numTags, args.emb_size, args.rnn_size, args.dropout_rate)
    train(data, tagger, args.num_epochs)