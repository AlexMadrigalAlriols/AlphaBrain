import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from pymongo import MongoClient
import wikipedia
import train

wikipedia.set_lang("es")

client = MongoClient("mongodb+srv://Madrigal:madrigalAlpha@alphainteligence.jwcwg.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db=client.words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('collection.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Alpha"
print("Let's chat! Type 'quit' to exit")

while True:
    sentence = input('You: ')

    if sentence == "quit":
        break

    palabra = tokenize(sentence)
    X = bag_of_words(palabra, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75 and tag != "Actions":
        for intent in intents["intents"]:
            if tag == 'Action1':
                barrer()
            elif tag == intent["tag"]:
                print(f"{bot_name}: {intent['responses'][0]}")
    else:
        print(f"{bot_name}: No entiendo, me podrias dar una respuesta a tu pregunta para la proxima?")
        palabra = input('You: ')
        train.trainBot()
        #response = dictionary.meaning(sentence)
        #print(wikipedia.summary(sentence, sentences=1))

        print(f"{bot_name}: ¿Quieres que guarde esa respuesta?")
        guardar = input('You: ')

        if guardar == "Si":
            print(f"{bot_name}: ¿A que tag pertenece?")
            tag = input('You: ')

            words = {
                'tag' : tag,
                'pattern' : [sentence],
                'responses' : [palabra]
            }

            result=db.reviews.insert_one(words)

            train.trainBot()
        else:
            print(f"{bot_name}: Okey!")

def barrer():
    print(f"{botname}: Barriendo!")

        
