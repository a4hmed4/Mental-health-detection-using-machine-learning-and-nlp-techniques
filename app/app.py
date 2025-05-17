import numpy as np
import streamlit as st
from keras.models import load_model, Model
from keras.layers import Input
import pickle
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

# --- 1. Load vocabulary and tokenizer ---
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('tokenizer_t.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Create word-to-token and token-to-word mappings using tokenizer's word_index
    encoder_word_to_token = tokenizer.word_index
    decoder_token_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    st.write("Vocabulary size:", len(vocab))
    st.write("Tokenizer word index size:", len(tokenizer.word_index))
except Exception as e:
    st.error(f"Error loading vocabulary: {e}")
    st.stop()

# --- 2. Load the trained model ---
try:
    training_model = load_model("training_model.h5")
    st.write("Model Summary:")
    training_model.summary(print_fn=st.write)
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
    st.write("Latent dimension:", latent_dim)

    # Use the original model's input dimensions
    decoder_inputs = Input(shape=(None, 172), name='decoder_inputs')  # Original shape from model
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

# --- 5. Prepare constants ---
num_encoder_tokens = 86  # Original encoder input size
num_decoder_tokens = 172  # Original decoder input size
max_encoder_seq_length = 20
max_decoder_seq_length = 20

# --- 6. Encoding and decoding functions ---
def encode_input_text(text):
    # Use the tokenizer to convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence
    padded = pad_sequences(sequence, maxlen=max_encoder_seq_length, padding='post')
    
    # Convert to one-hot encoding with original encoder size
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, token in enumerate(padded[0]):
        if token > 0 and token < num_encoder_tokens:  # Skip padding token (0) and ensure within bounds
            encoder_input_data[0, t, token] = 1.0
    
    return encoder_input_data

def decode_sequence(input_seq):
    # Get states from encoder
    states_value = encoder_model.predict(input_seq)

    # Initialize target sequence with original decoder size
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
    target_seq[0, 0, 0] = 1.0  # start token

    stop_condition = False
    decoded_sentence = ''
    max_attempts = 100  # Increased to allow for longer responses
    attempts = 0
    last_word = None
    min_words = 20  # Increased minimum words
    consecutive_repeats = 0
    max_repeats = 3

    while not stop_condition and attempts < max_attempts:
        attempts += 1
        
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Get top 10 predictions instead of 5
        top_indices = np.argsort(output_tokens[0, -1, :])[-10:][::-1]
        
        # Find first valid token index that's not the same as the last word
        sampled_token_index = None
        for idx in top_indices:
            if idx < len(vocab):  # Check if index is within vocabulary
                word = list(vocab)[idx]
                if word != last_word:  # Avoid repeating the same word
                    sampled_token_index = idx
                    consecutive_repeats = 0
                    break
                else:
                    consecutive_repeats += 1
        
        if sampled_token_index is None or consecutive_repeats >= max_repeats:
            break
            
        sampled_word = list(vocab)[sampled_token_index]
        
        # Add word to sentence
        if decoded_sentence:
            decoded_sentence += ' ' + sampled_word
        else:
            decoded_sentence = sampled_word
            
        last_word = sampled_word

        # Check stop conditions
        if (sampled_token_index == 1 or  # end token
            len(decoded_sentence.split()) > max_decoder_seq_length or  # max length
            (len(decoded_sentence.split()) >= min_words and attempts >= 20)):  # minimum response length
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    # If the response is too short, add a default message
    if len(decoded_sentence.split()) < min_words:
        decoded_sentence += " I understand this is a complex topic. Would you like to know more about this?"

    return decoded_sentence.strip()

# --- 7. Streamlit interface ---
st.title("Mental Health Chatbot")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.write(f"{message['role']}: {message['content']}")

# Get user input
user_input = st.text_input("You:")

if user_input:
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "You", "content": user_input})
        
        # Generate response using the generative model
        encoded_input = encode_input_text(user_input)
        response = decode_sequence(encoded_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "Bot", "content": response})
        
        # Display the new messages
        st.write(f"You: {user_input}")
        st.write(f"Bot: {response}")
        
    except Exception as e:
        st.error(f"Error processing input: {e}")
        st.session_state.chat_history.append({"role": "Bot", "content": "I'm sorry, I encountered an error. Please try again."})
