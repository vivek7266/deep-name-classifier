from __future__ import print_function

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential

maxlen = 30
labels = 2

input = pd.read_csv("origin_in.csv", header=None)
input.columns = ['name', 'n_or_f', 'namelen']

input.groupby('n_or_f')['name'].count()

names = input['name']
origin = input['n_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)

char_index = dict((c, i) for i, c in enumerate(vocab))

msk = np.random.rand(len(input)) < 0.8
train = input[msk]
test = input[~msk]


def tag_origin(n_or_f):
    result = []
    for elem in n_or_f:
        if elem == 'n':
            result.append([1, 0])
        else:
            result.append([0, 1])
    return result


def name_matrix(trunc_name_input, char_index_input, maxlen_input):
    result = []
    for i in trunc_name_input:
        tmp = [set_flag(char_index_input[j]) for j in str(i)]
        for k in range(0, maxlen_input - len(str(i))):
            tmp.append(set_flag(char_index_input["END"]))
        result.append(tmp)
    return result


def set_flag(i):
    tmp = np.zeros(56)
    tmp[i] = 1
    return tmp


trunc_train_name = [str(i)[0:maxlen] for i in train.name]
train_X = name_matrix(trunc_train_name, char_index, maxlen)
train_Y = tag_origin(train.n_or_f)

trunc_test_name = [str(i)[0:maxlen] for i in test.name]
test_X = name_matrix(trunc_test_name, char_index, maxlen)
test_Y = tag_origin(test.n_or_f)

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len_vocab)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1000
model.fit(train_X, train_Y, batch_size=batch_size, nb_epoch=10, validation_data=(test_X, test_Y))

# predcition
name = ["ac/dc", "chris brown", "elvis presley", "$hakira", "arijit singh", "shreya ghosal", "john abraham"]
trunc_name = [i[0:maxlen] for i in name]
X = name_matrix(trunc_name, char_index, maxlen)

pred = model.predict(np.asarray(X))

score, acc = model.evaluate(test_X, test_Y)
print('Test score:', score)
print('Test accuracy:', acc)

# save our model and data
model.save_weights('origin_model', overwrite=True)
train.to_csv("train_split.csv")
test.to_csv("test_split.csv")

evals = model.predict(test_X)
prob_m = [i[0] for i in evals]

out = pd.DataFrame(prob_m)
out['name'] = test.name.reset_index()['name']
out['n_or_f'] = test.m_or_f.reset_index()['n_or_f']

out.head(10)
out.columns = ['prob_n', 'name', 'actual']
out.head(10)
out.to_csv("origin_pred_out.csv")
