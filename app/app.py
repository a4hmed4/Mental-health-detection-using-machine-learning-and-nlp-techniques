import streamlit as st
import numpy as np
import re
import pickle
from keras.models import load_model, Model
from keras.layers import Input

# Load tokenizer
with open("tokenizer_t.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
index_word = tokenizer.index_word

num_tokens = len(word_index) + 1
latent_dim = 256
max_encoder_seq_length = 20
max_decoder_seq_length = 20

# Create token dictionaries
input_features_dict = word_index
target_features_dict = word_index
reverse_target_features_dict = index_word

# Load trained model
model = load_model("training_model.h5")

# Define encoder model
encoder_inputs = model.input[0]  # Encoder input
_, state_h_enc, state_c_enc = model.layers[2].output  # LSTM output
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Define decoder model
decoder_inputs = Input(shape=(None, num_tokens), name="decoder_input_inference")
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")

decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=[decoder_state_input_h, decoder_state_input_c]
)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs, state_h, state_c]
)

# Utility functions
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def encode_input(text):
    tokens = re.findall(r"[\w']+|[^\s\w]", text.lower())
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_tokens))
    for t, token in enumerate(tokens):
        if t >= max_encoder_seq_length:
            break
        index = input_features_dict.get(token)
        if index:
            encoder_input_data[0, t, index] = 1.
    return encoder_input_data

def decode_sequence(states_value):
    target_seq = np.zeros((1, 1, num_tokens))
    target_seq[0, 0, target_features_dict.get('<start>', 1)] = 1.

    decoded_sentence = ''

    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict.get(sampled_token_index, '')

        if sampled_token == '<end>':
            break

        decoded_sentence += ' ' + sampled_token
        target_seq = np.zeros((1, 1, num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence.strip()

def generate_answer(input_text):
    encoder_input_data = encode_input(input_text)
    states_value = encoder_model.predict(encoder_input_data)
    return decode_sequence(states_value)

# Streamlit UI
st.set_page_config(page_title="🧠 Mental Health Assistant", layout="centered")
st.title("🧠 Mental Health Q&A Generator")
st.write("Ask a mental health-related question below:")

user_input = st.text_input("Your Question")

if user_input:
    cleaned_input = clean_text(user_input)
    answer = generate_answer(cleaned_input)
    st.success(f"**Answer:** {answer}")
