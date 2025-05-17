import numpy as np
import streamlit as st
from keras.models import load_model, Model
from keras.layers import Input
import pickle
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load vocabulary and tokenizer ---
try:
    # Load tokenizer and vocabulary
    tokenizer_t = joblib.load('tokenizer_t.pkl')
    vocab = joblib.load('vocab.pkl')
    
    st.write("Vocabulary type:", type(vocab))
    st.write("First few vocabulary items:", list(vocab)[:10])
    st.write("Vocabulary size:", len(vocab))
    
    # Print tokenizer configuration
    st.write("Tokenizer configuration:")
    st.write("Tokenizer word index:", tokenizer_t.word_index)
    st.write("Tokenizer word counts:", tokenizer_t.word_counts)
    st.write("Tokenizer document count:", tokenizer_t.document_count)
    st.write("Tokenizer num words:", tokenizer_t.num_words)
    
except Exception as e:
    st.error(f"Error loading vocabulary: {e}")
    st.error(f"Error type: {type(e)}")
    st.error(f"Error details: {str(e)}")
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
    st.write("Encoder input shape:", encoder_inputs.shape)
    
    encoder_lstm = training_model.get_layer('lstm')
    st.write("LSTM layer config:", encoder_lstm.get_config())

    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)
    st.write("Encoder model summary:")
    encoder_model.summary(print_fn=st.write)
except Exception as e:
    st.error(f"Error building encoder model: {e}")
    st.stop()

# --- 4. Extract decoder model ---
try:
    latent_dim = state_h_enc.shape[-1]  # 256
    st.write("Latent dimension:", latent_dim)

    decoder_inputs = Input(shape=(None, len(vocab)), name='decoder_inputs')
    st.write("Decoder input shape:", decoder_inputs.shape)
    
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
    st.write("Decoder model summary:")
    decoder_model.summary(print_fn=st.write)
except Exception as e:
    st.error(f"Error building decoder model: {e}")
    st.stop()

# --- 5. Prepare constants ---
num_encoder_tokens = len(vocab)
num_decoder_tokens = len(vocab)
max_encoder_seq_length = 20
max_decoder_seq_length = 20

# --- 6. Encoding and decoding functions ---
def encode_input_text(text):
    # Print input text
    st.write("Input text:", text)
    
    # Use the tokenizer to convert text to sequence
    sequence = tokenizer_t.texts_to_sequences([text])
    st.write("Tokenized sequence:", sequence)
    
    # Pad the sequence
    padded = pad_sequences(sequence, maxlen=max_encoder_seq_length, padding='post')
    st.write("Padded sequence:", padded)
    
    # Convert to one-hot encoding
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, token in enumerate(padded[0]):
        if token > 0:  # Skip padding token (0)
            encoder_input_data[0, t, token] = 1.0
    
    st.write("Encoder input shape:", encoder_input_data.shape)
    st.write("Non-zero elements:", np.count_nonzero(encoder_input_data))
    return encoder_input_data

def decode_sequence(input_seq):
    st.write("Input sequence shape:", input_seq.shape)
    
    # Get states from encoder
    states_value = encoder_model.predict(input_seq)
    st.write("Encoder states shape:", [s.shape for s in states_value])

    # Initialize target sequence
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
    target_seq[0, 0, 0] = 1.0  # start token
    st.write("Initial target sequence shape:", target_seq.shape)

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # Print shapes before prediction
        st.write("Shapes before decoder prediction:")
        st.write("Target seq shape:", target_seq.shape)
        st.write("States shapes:", [s.shape for s in states_value])
        
        # Predict with correct input shapes
        try:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            st.write("Output tokens shape:", output_tokens.shape)
            st.write("Output tokens sample:", output_tokens[0, -1, :10])  # Print first 10 probabilities
        except Exception as e:
            st.error(f"Error in decoder prediction: {str(e)}")
            st.error(f"Expected input shape: (None, None, {num_decoder_tokens})")
            st.error(f"Actual input shape: {target_seq.shape}")
            raise e

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        st.write("Sampled token index:", sampled_token_index)
        
        sampled_word = decoder_token_to_word.get(sampled_token_index, f"[UNK_{sampled_token_index}]")
        st.write("Sampled word:", sampled_word)
        
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
