from tkinter import *
import tkinter
import random
import json
from train_chatbot import sequential_model
import torch
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
lemmatizer = WordNetLemmatizer()

# load the trained stats
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = sequential_model()
model.load_state_dict(torch.load('model.net'))

# handle input sentence


def clean_input(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(token.lower())
              for token in tokens if token != '!' and token != '?']
    return tokens


def to_bow(sentence, words):
    tokens = clean_input(sentence)
    bag = [0] * len(words)
    for i in range(0, len(tokens)):
        if tokens[i] in words:
            bag[words.index(tokens[i])] = 1

    bag = torch.FloatTensor(np.array(bag))
    return bag


def predict_target(sentence, model, temperature=1.5):
    inp = to_bow(sentence, words)
    model.eval()
    out = model(inp)
    # Apply temperature
    results = [[i, r] for i, r in enumerate(out)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
#     print(results)
    for r in results:
        return_list.append(
            {"intent": classes[r[0]], "probability": str(r[1].detach().numpy())})
    return return_list


def respond(sentence):
    pred_int = predict_target(sentence, model)
    ans = pred_int[0]['intent']
    for intent in intents['intents']:
        if ans == intent['tag']:
            return random.choice(intent['responses'])


# Creating GUI with tkinter


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = respond(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("AI chatbox")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
