from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz  # For fuzzy matching

# Download required NLTK tokenizer
nltk.download('punkt')
stemmer = PorterStemmer()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model and data
model = tf.keras.models.load_model("chatbot_model.h5")

with open("vocab.json") as f:
    vocab = json.load(f)
with open("labels.json") as f:
    labels = json.load(f)
with open("responses.json") as f:
    responses = json.load(f)
with open("intents.json") as f:
    intent_data = json.load(f)

# Tokenization and BOW
def tokenize(sentence):
    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]

def bag_of_words(sentence, vocab):
    bag = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            index = vocab.index(word)
            bag[index] = 1
    return bag

# Fuzzy match user input to best examples
def get_best_intent_fuzzy(user_input):
    best_score = 0
    best_intent = None
    for name in intent_data["intents"]:
        for examples in name["examples"]:
            score = fuzz.ratio(user_input.lower(), examples.lower())
            if score > best_score:
                best_score = score
                best_intent = name
    return best_intent, best_score

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    print("User input:", user_input)

    # Fuzzy match
    matched_intent, score = get_best_intent_fuzzy(user_input)
    print(f"Fuzzy match score: {score}")

    if matched_intent and score > 70:
        predicted_label = matched_intent["name"]
        print(f"Predicted name: {predicted_label}")
        answer = random.choice(responses[predicted_label])
    else:
        answer = "I'm sorry, I didn't understand that. Can you try asking in a different way?"

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
