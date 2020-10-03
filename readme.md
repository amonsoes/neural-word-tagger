## Neural Word Tagger

### This is an algorithm to train and use an Long Short Term Memory Neural Network to perform word tagging on a sentence level

Usage: **through CLI**

**install all requirements:**

`pip3 install -r requirements.txt`

**training:**

file: rnn-training.py

positional arguments:

- train data (training data is in ./data/)
- dev data (development data is in ./data/)
- path to serialize data class and tagger paramters (/path/to/tagger - will result in tagger.io, tagger.rnn)

other arguments:

- --num_epochs: number of epochs
- --emb_size: dimensionality of the embedding layer
- --rnn_size: dimensionality of the hidden RNN-vector
- --dropout_rate: introduces slight training error to fight overfitting
- --learning_rate: sets learning rate of the optimizer
- --gpu: set if training should be done on a GPU(recommended) or on CPU

example:

`python3.7 rnn-training.py ./data/train.tagged ./data/dev.tagged ./saves/tagger --gpu True --num_epochs 3 --num_words 10000 --emb_size 100 --rnn_size 200 --dropout_rate 0.5 --learning_rate 0.01`


**annotate:**

file: rnn-annotate.py

arguments:

- --path_param: set path to the serialized files <path_param>.rnn <path_param>.io
- --sent_param: set path to sentences to annotate
- --gpu: will change the device of the NN to either GPU or CPU

example:

`python3.7 rnn-annotate.py --path_param ./saves/tagger --path_sents ./test_data/annotate_set.txt --gpu False
`


Best training accuracy observed: 95,8%
Hyperparamerter used:

- --num_epochs: 3
- --emb_size: 300
- --rnn_size: 500
- --dropout_rate: 0.4
- --learning_rate: 0.001
- --gpu: True


