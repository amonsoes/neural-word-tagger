import random
import torch
import argparse

from sys import argv
from classes import tagger as tg

parser = argparse.ArgumentParser(description='Set hyperparams for tagger RNN')

#positional args
parser.add_argument('parfile', input=str, help='set file to load/dump data attributes. Needs suffix')

#optional args
parser.add_argument('--num_epochs', input=int, help='set the number of epochs of the training')
parser.add_argument('--num_words', input=int, help='set the number of words. This will impact the training speed, but also affects quality')
parser.add_argument('--emb_size', input=int, help='set the number of dimensions of the embedding matrix. This will impact the training speed, but also affects quality')
parser.add_argument('--rnn_size', input=int, help='set the number of dimensions of the LSTM vector. This will impact the training speed, but also affects quality')
parser.add_argument('--dropout_rate', input=float, help='set the dropout rate')
parser.add_argument('--learning_rate', input=float, help='set the learning rate of the optimizer')

parser.parse_args()

def train(data, tagger, numEpochs):
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
    
    NUMWORDS = args.num_words
    EMBSIZE = args.emb_size
    RNNSIZE = args.rnn_size
    NUMEPOCHS = args.num_epochs
    DO_RATE = args.dropout_rate
    ftrain = "./data/train.tagged"
    fdev = "./data/dev.tagged"
    
    data = tg.Data(ftrain, fdev, NUMWORDS)
    tagger = tg.TaggerModel(NUMWORDS, data.numTags, EMBSIZE, RNNSIZE, DO_RATE)
    train(data, tagger, 5)