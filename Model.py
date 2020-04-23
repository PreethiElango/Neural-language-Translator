from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional
import numpy as np
import json
import pandas as pd
from keras.models import load_model

def training():
    eng_vocab = set()
    shake_vocab = set()
    english = []
    shakespear = []
    batch_size = 32  
    epochs = 15
    latent_dim = 256  
    num_samples = 17000  
    data_path = 'data/Training.txt'      
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        target_text, input_text = line.split('\t')
        target_text = '\t' + target_text + '\n'
        english.append(input_text)
        shakespear.append(target_text)
        for char in input_text:
            if char not in eng_vocab:
                eng_vocab.add(char)
        for char in target_text:
            if char not in shake_vocab:
                shake_vocab.add(char)

    eng_vocab = sorted(list(eng_vocab))
    shake_vocab = sorted(list(shake_vocab))
    en_tokens = len(eng_vocab)
    de_tokens = len(shake_vocab)
    
    en_length = max([len(txt) for txt in english])
    de_length = max([len(txt) for txt in shakespear])
    input_token_index = dict([(char, i) for i, char in enumerate(eng_vocab)])
    target_token_index = dict([(char, i) for i, char in enumerate(shake_vocab)])
    
    en_encoding = np.zeros((len(english), en_length, en_tokens),dtype='float32')
    de_in_encoding = np.zeros((len(english), de_length, de_tokens),dtype='float32')
    de_op_encoding = np.zeros((len(english), de_length, de_tokens),dtype='float32')
    
    for i, (input_text, target_text) in enumerate(zip(english, shakespear)):
        for t, char in enumerate(input_text):
            en_encoding[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            de_in_encoding[i, t, target_token_index[char]] = 1.
            if t > 0:
                de_op_encoding[i, t - 1, target_token_index[char]] = 1.
                
    encoder_inputs = Input(shape=(None, en_tokens))
    encoder = LSTM(latent_dim, dropout=0.5, return_state=True)
    a, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, de_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(de_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([en_encoding, de_in_encoding], de_op_encoding,batch_size=batch_size,epochs=epochs,
              validation_split=0.2)
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save('encoder.h5')
    print("model saved")
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    decoder_model.save('decoder.h5')
    print("model saved")
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    with open('de_tokens.json','w') as fp:
        json.dump(de_tokens,fp)
    with open('target_token_index.json','w') as fp:
        json.dump(target_token_index,fp)
    with open('reverse_target_char_index.json','w') as fp:
        json.dump(reverse_target_char_index,fp)
    with open('de_length.json','w') as fp:
        json.dump(de_length,fp)
    with open('input_token_index.json','w') as fp:
        json.dump(input_token_index,fp)
        
    
def decode_sequence(input_seq):
    with open('de_tokens.json','r') as fp:
        de_tokens=json.load(fp)
    with open('target_token_index.json','r') as fp:
        target_token_index=json.load(fp)
    with open('reverse_target_char_index.json','r') as fp:
        reverse_target_char_index=json.load(fp)
    with open('de_length.json','r') as fp:
        de_length=json.load(fp)
    with open('de_tokens.json','r') as fp:
        de_tokens=json.load(fp)
    encoder_model = load_model('encoder.h5')
    decoder_model = load_model('decoder.h5')
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, de_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[str(sampled_token_index)]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > de_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, de_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence

def encode_sequence(string):
    with open('input_token_index.json','r') as fp:
        input_token_index=json.load(fp)
    test = np.zeros((1, 28, 71),dtype='float32')
    for i, input_text in enumerate(string):
        for t, char in enumerate(input_text):
            test[i, t, input_token_index[char]] = 1.
    decoded_sentence = decode_sequence(test)
    print(decoded_sentence)
