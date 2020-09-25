import random
import torch
import argparse

from sys import argv
from classes import tagger as tg

def optimize(x, y, optimizer, model, data):
    optimizer.zero_grad()
    output = model(torch.LongTensor(data.words2IDs(x)))
    loss = torch.nn.CrossEntropyLoss().cuda()
    loss_output = loss(output, torch.LongTensor(data.tags2IDs(y)).cuda())
    loss_output.backward()
    optimizer.step()

def dev_evaluate(x, y, model, data, total_tagged, sum_corr):
    total_tagged += len(y)
    output = tagger(torch.LongTensor(data.words2IDs(x)))
    sum_corr += sum([1 for ix,iy in zip(output,torch.LongTensor(data.tags2IDs(y))) if torch.argmax(ix).item() == iy.item()])
    accuracy = sum_corr / total_tagged
    return accuracy

def check_accuracy(accuracy, best_current_accuracy, model):
    if accuracy > best_current_accuracy:
        best_current_accuracy = accuracy
        print("====\nBEST ACCURACY CHANGED : {}\n====".format(best_current_accuracy))
        torch.save(model, args.parfile+'.rnn')
    return best_current_accuracy
    
def train(data, tagger, numEpochs, optimizer):
    tagger.train()
    if args.gpu:
        tagger.cuda()
        tagger.to(tagger.device)
    best_current_acc = 0.0
    for epoch in range(numEpochs):
        for x, y in data.trainSentences:
            optimize(x, y, optimizer, tagger, data)
        random.shuffle(data.trainSentences)
        tagger.train(mode=False)
        total_tagged, sum_corr = 0, 0
        for x, y in data.devSentences:
            accuracy = dev_evaluate(x, y, tagger,data, total_tagged, sum_corr)
        best_current_acc = check_accuracy(accuracy, best_current_acc, tagger)

if __name__ == '__main__':
    
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
    parser.add_argument('--gpu', type=tg.str2bool, nargs='?', const=True, default=False, help='set True if cuda-able GPU is available. Else set False')

    args = parser.parse_args()
    
    print('Initisalizing training...\n\n Parameters:\n')
    print('  NUMWORDS : {}\n  EMBSIZE : {}\n  RNNSIZE : {}\n  NUMEPOCHS :{}\n  DO_RATE :{}\n  L_RATE : {}\n  CUDA : {}\n\n'.format(args.num_words, args.emb_size, args.rnn_size, args.num_epochs, args.dropout_rate, args.learning_rate, args.gpu))

    dataset = tg.Data(args.trainfile, args.devfile, args.num_words)
    tagger = tg.TaggerModel(args.num_words, dataset.numTags, args.emb_size, args.rnn_size, args.dropout_rate)
    train(dataset, tagger, args.num_epochs, torch.optim.SGD(tagger.parameters(), lr=args.learning_rate))