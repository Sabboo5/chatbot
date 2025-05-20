import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import difflib

# Ensure NLTK data is available
nltk.download('punkt')

# Load assets
model = load_model("chatbot_model.h5")
with open("vocab.json", "r") as f:
    vocab = json.load(f)
with open("labels.json", "r") as f:
    labels = json.load(f)
with open("responses.json", "r") as f:
    responses_data = json.load(f)

# Preprocessing
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stems = [stemmer.stem(w) for w in tokens]
    bag = [1 if word in stems else 0 for word in vocab]
    return np.array(bag)

# Optional: city name detection (can be extended)
city_keywords = ["axum", "lalibela", "addis ababa", "gondar", "bahir dar", "simien mountains", "sof omar caves"]

def extract_city_name(text):
    for city in city_keywords:
        if f" {city} " in f" {text.lower()} ":
            return city
    return None

# Fuzzy match before model use
def get_best_match(user_input, responses_data):
    highest_score = 0
    matched_intent = None
    for entry in responses_data:
        for examples in entry.get("examples", []):
            score = difflib.SequenceMatcher(None, user_input.lower(), examples.lower()).ratio()
            if score > highest_score:
                highest_score = score
                matched_intent = entry.get("name")
    return matched_intent, highest_score

# Chat loop
print("🤖 Ethiopia Travel Chatbot is ready! (Type 'quit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("👋 Goodbye!")
        break

    city = extract_city_name(user_input)
    if city:
        print(f"📍 You're asking about {city.title()}.")

    # Try fuzzy match first
    matched_intent, similarity = get_best_match(user_input, responses_data)
    if similarity > 0.85:
        intent = matched_intent
        print(f"🧠 Matched by similarity (score: {similarity:.2f}): {intent}")
    else:
        # Use model if no fuzzy match
        input_data = preprocess(user_input)
        prediction = model.predict(np.array([input_data]))[0]
        max_index = np.argmax(prediction)
        confidence = prediction[max_index]
        intent = labels[max_index]
        print(f"🤖 Predicted intent by model: {intent} (confidence: {confidence:.2f})")

        if confidence < 0.7:
            print("Bot: Sorry, I didn't understand that. Can you rephrase?")
            continue

    # Find response
    response_list = None
    for entry in responses_data:
        if entry.get("name", "").lower() == intent.lower():
            response_list = entry.get("responses", [])
            break

    if response_list:
        if city:
            filtered = [r for r in response_list if city.lower() in r.lower()]
            response = random.choice(filtered if filtered else response_list)
        else:
            response = random.choice(response_list)
        print("Bot:", response)
    else:
        print(f"Bot: No responses found for intent '{intent}'.")
