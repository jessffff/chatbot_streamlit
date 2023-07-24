
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import streamlit as st

# Load the trained model, tokenizer, and label encoder using pickle
with open('chatbot_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)



model = tf.keras.models.load_model("chatbot_model_tf")
# Function to get the chatbot response


def chatbot_response(user_input):
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_sequence = pad_sequences(
        input_sequence, maxlen=max_sequence_length, padding='post')
    predicted_tag_index = np.argmax(model.predict(input_sequence))
    predicted_tag = label_encoder.inverse_transform([predicted_tag_index])[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])


# Load and preprocess the data from the JSON file
with open("intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []
responses = []

for intent in data["intents"]:
    patterns.extend(intent["patterns"])
    tags.extend([intent["tag"]] * len(intent["patterns"]))
    responses.extend(intent["responses"])

# Tokenize the input user patterns
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1
input_sequences = tokenizer.texts_to_sequences(patterns)
max_sequence_length = max(len(seq) for seq in input_sequences)


# Create the Streamlit application


def main():
    st.title("Simple Chatbot with Streamlit")

    user_input = st.text_input("You: ", "")

    if st.button("Ask"):
        if user_input:
            response = chatbot_response(user_input)
            st.text(f"Bot: {response}")
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
