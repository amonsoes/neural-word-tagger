import torch
import json
import argparse

from torch import nn
from collections import Counter

parser = argparse.ArgumentParser(description='Set hyperparams for tagger RNN')
parser.add_argument('--parfile', type=str, help='set file to load/dump data attributes. Needs suffix')
parser.add_argument('--train', type=str, help='set training file')
parser.add_argument('--dev', type=str, help='set dev file')

args = parser.parse_args()


class Data:
    
    
    def __init__(self, *args):
        if len(args) == 1:
            self.init_test(*args)
        else:
            self.init_train(*args)
    
    def init_test(self, *args):
        with open(args.parfile, 'r',) as f:
            self.tag_id, self.word_id = json.load(f)
        self.numTags = len(self.tag_id.keys())
    
    def init_train(self,trainFile, devFile, numWords):
        self.trainSentences = self.readData(trainFile, True, numWords)
        self.devSentences = self.readData(devFile, False)
        self.numTags = len(self.tag_id)
        
    def readData(self, file, train, numwords=None):
        if train:
            wordFreq = Counter()
            tag_set = set()
        with open(file, 'r', encoding='utf-8') as f:
            sents, words, tags = [], [], []
            for line in f:
                if line == '\n':
                    sents.append((words, tags))
                    words, tags = [], []
                else:
                    word, tag = line.strip().split('\t')
                    words.append(word)
                    tags.append(tag)
                    if train:
                        wordFreq[word] += 1
                        tag_set.add(tag)
        if train:
            self.word_id = {w[0] : e+1 for e,w in enumerate(wordFreq.most_common(numwords))}
            #self.word_id['UNK'] will be implicitly 0
            self.tag_id = {t : e+1 for e,t in enumerate(tag_set)}
            self.tag_id['UNK'] = 0
            self.id_tag = { i : t for t,i in self.tag_id.items()}
            return sents
        return sents
    
    def sentences(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            sent = []
            for line in f:
                if line == '\n':
                    yield sent
                    sent = []
                else:
                    word = line.strip()
                    sent.append(word)
                    
    
    def words2IDs(self, words):
        return [self.word_id.get(w, 0) for w in words]
        
    def tags2IDs(self, tags):
        return [self.tag_id[t] for t in tags]
    
    def IDs2tags(self, bestTagIDs):
        return [self.id_tag[i] for i in bestTagIDs]
    
    def store_parameters(self, path):
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump((self.word_id, self.tag_id), f)
        
    def run_test(self):
        for words, tags in self.trainSentences:
            print(self.words2IDs(words))
            print(self.tags2IDs(tags))
        for words, tags in self.devSentences:
            print(words, " : " , tags )
            print(self.words2IDs(words))
        print(self.IDs2tags([2,5,12]))
            
        print("\n====== Data class functionality successfully tested ======\n")
        

class TaggerModel(nn.Module):
    
    def __init__(self, numWords, numTags, embSize, rnnSize, dropoutRate):
        super(TaggerModel, self).__init__()
        self.embedding_layer = nn.Embedding(numWords+1, embSize)
        self.lstm = nn.LSTM(embSize, rnnSize, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropoutRate)
        self.fc = nn.Linear(rnnSize*2, numTags)
        
    def forward(self, input):
        embeddings = self.embedding_layer(input)
        do_embeddings = self.dropout(embeddings)
        output, _ = self.lstm(torch.unsqueeze(do_embeddings, dim=0))
        do_vector = self.dropout(torch.squeeze(output, dim=0))
        output = self.fc(do_vector)
        return output
        
        
def run_test():
    
    NUMWORDS = 10000
    EMBSIZE = 200
    RNNSIZE = 200
    
    data = Data(args.parfile)
    tagger = TaggerModel(NUMWORDS, data.numTags, EMBSIZE, RNNSIZE, 0.1)
    
    data.store_parameters(args.parfile+'.io')
    
    print("\n====== Models successfully initialized ======\n")
    
    data.run_test()
    print(tagger(torch.LongTensor([2,5,12])).size())
    
    print("\n====== Tagger class functionality successfully tested ======\n")
    
if __name__ == '__main__':
    run_test()
    