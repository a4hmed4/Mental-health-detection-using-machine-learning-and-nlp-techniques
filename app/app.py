import numpy as np
import streamlit as st
from keras.models import load_model, Model
from keras.layers import Input
import pickle

# --- 1. Load vocabulary and tokenizer ---
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('tokenizer_t.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Create word-to-token and token-to-word mappings
    encoder_word_to_token = {word: idx for idx, word in enumerate(vocab)}
    decoder_token_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    st.write("Vocabulary size:", len(vocab))
except Exception as e:
    st.error(f"Error loading vocabulary: {e}")
    st.stop()

# --- 2. Load the trained model ---
try:
    training_model = load_model("training_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. Extract encoder model ---
try:
    encoder_inputs = training_model.input[0]  # input_1 with shape (None, None, 86)
    encoder_lstm = training_model.get_layer('lstm')

    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)
except Exception as e:
    st.error(f"Error building encoder model: {e}")
    st.stop()

# --- 4. Extract decoder model ---
try:
    latent_dim = state_h_enc.shape[-1]  # 256

    decoder_inputs = Input(shape=(None, 172), name='decoder_inputs')  # input_2 shape
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = training_model.get_layer('lstm_1')
    decoder_dense = training_model.get_layer('dense')

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
except Exception as e:
    st.error(f"Error building decoder model: {e}")
    st.stop()

# --- 4. Prepare dictionaries and constants ---
num_encoder_tokens = len(vocab)  # Use actual vocabulary size
num_decoder_tokens = len(vocab)  # Use actual vocabulary size
max_encoder_seq_length = 20
max_decoder_seq_length = 20

# Update the encoding function
def encode_input_text(text):
    tokens = text.lower().split()
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, token in enumerate(tokens):
        if t >= max_encoder_seq_length:
            break
        token_index = encoder_word_to_token.get(token)
        if token_index is not None:
            encoder_input_data[0, t, token_index] = 1.0
    return encoder_input_data

# Update the decoding function
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
    target_seq[0, 0, 0] = 1.0  # start token

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = decoder_token_to_word.get(sampled_token_index, f"[UNK_{sampled_token_index}]")
        decoded_sentence += ' ' + sampled_word

        if sampled_token_index == 1 or len(decoded_sentence.split()) > max_decoder_seq_length:  # 1 is end token
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence.strip()

# --- 7. Streamlit interface ---
st.title("Mental Health Chatbot")

user_input = st.text_input("You:")

if user_input:
    encoded_input = encode_input_text(user_input)
    response = decode_sequence(encoded_input)
    st.write("Bot:", response)
