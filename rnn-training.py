import random
import torch

from sys import argv
from classes import tagger as tg

parser = argparse.ArgumentParser(description='Set hyperparams for tagger RNN')
parser.add_argument('parfile', input=string, help='set file to load/dump data attributes. Needs suffix')
parser.parse_args()

def train(data, tagger, numEpochs):
    optimizier = torch.optim.Adam(tagger.parameters(), lr=0.001)
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
    
    NUMWORDS = 10000
    EMBSIZE = 200
    RNNSIZE = 200
    NUMEPOCHS = 20
    ftrain = "./data/train.tagged"
    fdev = "./data/dev.tagged"
    
    data = tg.Data(ftrain, fdev, NUMWORDS)
    tagger = tg.TaggerModel(NUMWORDS, data.numTags, EMBSIZE, RNNSIZE, 0.1)
    train(data, tagger, 5)