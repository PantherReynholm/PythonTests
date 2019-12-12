from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.layers import Dropout, Dense, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

file = open('epdf.pub_the-way-of-kings.txt', 'r+', encoding="utf8")
file = file.read()


def process_training_text(reduce_data):
    way_of_kings_words = text_to_word_sequence(file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(way_of_kings_words)
    tokenized_text = tokenizer.texts_to_sequences(way_of_kings_words)
    tokenized_text = tokenized_text[:int(np.floor(len(tokenized_text) * reduce_data))]
    tokenized_text = np.squeeze(tokenized_text)
    return tokenized_text


def process_test_text(reduce_data, final_data):
    way_of_kings_words = text_to_word_sequence(file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(way_of_kings_words)
    tokenized_text = tokenizer.texts_to_sequences(way_of_kings_words)
    tokenized_text = tokenized_text[int(np.floor(len(tokenized_text) * reduce_data)): int(np.floor(len(tokenized_text) * final_data))]
    tokenized_text = np.squeeze(tokenized_text)
    number_to_word = tokenizer.word_index
    return tokenized_text, number_to_word


def create_series_of_words(token_text, seq_len):
    series_matrix = []
    for i in tqdm(range(len(token_text) - seq_len - 1)):
        sub_matrix = token_text[i: i + seq_len]
        value = sub_matrix[-1]
        series_matrix.append([sub_matrix, value])
    return series_matrix


def form_input_matrix(tokenized_text):

    nn_input_matrix = []
    category_number = []
    for input_matrix, label in tqdm(tokenized_text):
        nn_input_matrix.append(input_matrix[:-1])
        category_number.append(label)
    category_number = to_categorical(category_number)

    nn_input_matrix = np.array(nn_input_matrix)
    nn_input_matrix = nn_input_matrix.reshape(len(nn_input_matrix), len(nn_input_matrix[0]), 1)

    return nn_input_matrix, category_number
