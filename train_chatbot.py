from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pickle
import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# loading data
words = []
tags = []
corpus = []
intents = json.loads(open('intents.json').read())
'''
each entry in intents is in the form of 

{"tag": "...",
"patterns": ["...", "...", "...","...","...", "...", "..."], #the sentences of inputs
"responses": ["...", "...", "..."], #the desired form of outputs
"context": ["...", "..."]
}
'''

# pre-processing data in intents
for intent in intents['intents']:
    # tokenize each sentence in pattern
    for pattern in intent['patterns']:
        token = nltk.word_tokenize(pattern)
        words += token

        # add the correspondence of {tag: token} to corpus
        tag = intent['tag']
        corpus.append((tag, token))

        # add tag to tags
        if tag not in tags:
            tags.append(tag)


# lemmatize the words and remove the duplicate
words = [lemmatizer.lemmatize(token.lower())
         for token in words if token != '!' and token != '?']
# sort the words and tags
words = sorted(list(set(words)))
classes = sorted(list(set(tags)))

# save them in pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

dictionary = []
for word in words:
    for letter in word:
        if letter not in dictionary:
            dictionary.append(letter)
dictionary = sorted(dictionary)


def target_encoding(tag):
    output = [0] * len(classes)
    idx = classes.index(tag)
    output[idx] = 1
    return output


# create training and test data
training = []

for corpora in corpus:
    # encode the input patterns
    target = corpora[0]
    sentence = corpora[1]
    sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence]
    # create one-hot encoding of words in sentence
    bag = []
    for word in words:
        bag.append(1) if word in sentence else bag.append(0)
    # create one-hot encoding of target
    target = target_encoding(target)
    training.append([bag, target])
random.shuffle(training)
train_x = np.array([training[i][0] for i in range(len(training))])
train_y = np.array([training[j][1] for j in range(len(training))])
# customize dataset


class dataset(Dataset):

    def __init__(self):
        self.inp = train_x
        self.out = train_y

        self.inp = torch.FloatTensor(self.inp)
        self.out = torch.FloatTensor(self.out)

    def __getitem__(self, idx):
        return self.inp[idx], self.out[idx]

    def __len__(self):
        return len(self.inp)

# create the neural networks


class sequential_model(nn.Module):
    def __init__(self) -> None:
        super(sequential_model, self).__init__()
        self.fc1 = nn.Linear(len(train_x[0]), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(train_y[0]))
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        output = F.log_softmax(x)
        return output


def train(net, data, epochs=5, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for e in range(epochs):
        for i_batch, batch_data in enumerate(data):
            inp, target = batch_data
            net.zero_grad()
            output = net(inp)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            if i_batch % print_every == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}/{}...".format(i_batch, len(data)),
                      "Loss: {:.4f}...".format(loss.item()),
                      )


model = sequential_model()
data = dataset()
dataLoader = DataLoader(data, batch_size=3, shuffle=True)

n_epochs = 300
train(model, dataLoader, epochs=n_epochs)

model_name = 'model.net'
with open(model_name, 'wb') as f:
    torch.save(model.state_dict(), f)
