import json
import numpy as np
import tensorflow as tf
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk

nltk.download('punkt')

# Load structured data
with open("intents.json") as file:
    data = json.load(file)

stemmer = PorterStemmer()

training_sentences = []
training_labels = []
all_labels = []
all_words = []

# Prepare data using both examples and tags
for intent in data["intents"]:
    for examples in intent["examples"]:
        tokens = nltk.word_tokenize(examples.lower())
        stemmed = [stemmer.stem(word) for word in tokens if word.isalnum()]
        all_words.extend(stemmed)
        training_sentences.append(" ".join(stemmed))
        training_labels.append(intent["name"])
        if intent["name"] not in all_labels:
            all_labels.append(intent["name"])

# Encode the labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

# Create bag of words
vocab = sorted(set(all_words))

def bag_of_words(sentence, vocab):
    """ Convert sentence into a bag of words representation """
    bag = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            index = vocab.index(word)
            bag[index] = 1
    return np.array(bag)

X_train = np.array([bag_of_words(sent, vocab) for sent in training_sentences])
y_train = np.array(training_labels_encoded)

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(vocab),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(training_labels)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=1)

# Save the model and necessary files
model.save("chatbot_model.h5")
with open("vocab.json", "w") as f:
    json.dump(vocab, f)
with open("labels.json", "w") as f:
    json.dump(lbl_encoder.classes_.tolist(), f)
with open("responses.json", "w") as f:
    responses = {intent["name"]: intent["responses"] for intent in data["intents"]}
    json.dump(responses, f)

print("✅ Model training completed and files saved.")
