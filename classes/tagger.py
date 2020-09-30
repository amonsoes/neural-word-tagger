import torch
import json
import argparse

from torch import nn
from collections import Counter

from . import tools

class Data:
    
    def __init__(self, *args):
        if len(args) == 1:
            self.init_test(*args)
        else:
            self.init_train(*args)
    
    def init_test(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            self.word_id, self.tag_id = json.load(f)
        self.numTags = len(self.tag_id.keys())
        self.id_tag = { v : k for k, v in self.tag_id.items()}
    
    def init_train(self,trainFile, devFile, numChar):
        self.trainSentences = self.readData(trainFile, True, numChar)
        self.devSentences = self.readData(devFile, False)
        self.numTags = len(self.tag_id)
        
    def readData(self, file, train, numchars=None):
        if train:
            charFreq = Counter()
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
                        for char in word:
                            charFreq[char] += 1
                        tag_set.add(tag)
        if train:
            self.char_id = {w : e+1 for e,w in enumerate(charFreq) if charFreq[w] > 1}
            #self.word_id['UNK'] will be implicitly 0
            self.tag_id = {t : e+1 for e,t in enumerate(tag_set)}
            # unknown tag will be implicitly 0 self.tag_id['UNK'] = 0
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
                    
    def words2IDvecs(self, words):
        pref_suff = [(w[:10],w[len(w)-10:]) if len(w) > 10 else (w+' '*(10-len(w)), ' '*(10-len(w))+w) for w in words]
        p_mat, s_mat = [], []
        for pref, suff in pref_suff:
            p_mat.append([self.char_id.get(char, 0) for char in pref[::-1]])
            s_mat.append([self.char_id.get(char,0) for char in suff])
        return torch.LongTensor(p_mat), torch.LongTensor(s_mat)
        
    def tags2IDs(self, tags):
        return [self.tag_id.get(t, 0) for t in tags]
    
    def IDs2tags(self, bestTagIDs):
        return [self.id_tag.get(i, 0) for i in bestTagIDs]
    
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
    
    def __init__(self, numChars, numTags, embSize, rnnSize, dropoutRate, has_gpu):
        super(TaggerModel, self).__init__()
        self.embedding = nn.Embedding(numChars+1, embSize)
        self.char_lstm = nn.LSTM(embSize ,rnnSize, batch_first=True)
        self.lstm = nn.LSTM(rnnSize*2, rnnSize, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropoutRate)
        self.fc = nn.Linear(rnnSize*2, numTags+1)
        self.device = torch.device("cuda" if has_gpu else "cpu")
        
    def forward(self, input):
        if self.device.type == 'cuda':
            input = input.cuda()
            embeddings = self.embedding_layer(input).cuda()
            do_embeddings = self.dropout(embeddings).cuda()
            output, _ = self.lstm(torch.unsqueeze(do_embeddings, dim=0))
            do_vector = self.dropout(torch.squeeze(output.cuda(), dim=0)).cuda()
            output = self.fc(do_vector).cuda()
        else:
            embeddings = self.embedding_layer(input)
            do_embeddings = self.dropout(embeddings)
            output, _ = self.lstm(torch.unsqueeze(do_embeddings, dim=0))
            do_vector = self.dropout(torch.squeeze(output, dim=0))
            output = self.fc(do_vector)
        return output

def store_data(trainfile, devfile, numwords):
    data = Data(trainfile, devfile, numwords)
    data.store_parameters(tools.handle_path_coll(args.parfile+'.io'))
        
def run_test(numwords):
    
    EMBSIZE = 200
    RNNSIZE = 200
    
    data = Data(args.parfile)
    tagger = TaggerModel(numwords, data.numTags, EMBSIZE, RNNSIZE, 0.1)
    
    print("\n====== Models successfully initialized ======\n")
    
    data.run_test()
    print(tagger(torch.LongTensor([2,5,12])).size())
    
    print("\n====== Tagger class functionality successfully tested ======\n")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Set hyperparams for tagger RNN')
    parser.add_argument('--parfile', type=str, help='set file to load/dump data attributes. Needs suffix')
    parser.add_argument('--train', type=str, help='set training file')
    parser.add_argument('--dev', type=str, help='set dev file')
    parser.add_argument('--gpu', type=tools.str2bool, nargs='?', const=True, default=False, help='set True if cuda-able GPU is available. Else set False')
    parser.add_argument('--numwords', type=int, help='set number of known words')
    
    args = parser.parse_args()
    
    store_data(args.train, args.dev, args.numwords)
    