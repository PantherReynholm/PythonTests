from keras.layers import Dropout, Dense, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from text_processor import process_training_text, create_series_of_words, form_input_matrix
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.layers import Dropout, Dense, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm import tqdm

SEQ_LEN = 61
EPOCHS = 50
BATCH_SIZE = 64
reduce_data = 0.1

file = open('epdf.pub_the-way-of-kings.txt', 'r+', encoding="utf8")
file = file.read()

way_of_kings_words = text_to_word_sequence(file)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(way_of_kings_words)
tokenized_text = tokenizer.texts_to_sequences(way_of_kings_words)
tokenized_text = tokenized_text[:int(np.floor(len(tokenized_text) * reduce_data))]
tokenized_text = np.squeeze(tokenized_text)

series_matrix = []
for i in tqdm(range(len(tokenized_text) - SEQ_LEN - 1)):
    sub_matrix = tokenized_text[i: i + SEQ_LEN]
    value = sub_matrix[-1]
    series_matrix.append([sub_matrix, value])

nn_input_matrix = []
category_number = []
for input_matrix, label in tqdm(series_matrix):
    nn_input_matrix.append(input_matrix[:-1])
    category_number.append(label)

nn_input_matrix = np.array(nn_input_matrix)
nn_input_matrix = nn_input_matrix.reshape(nn_input_matrix.shape[0], SEQ_LEN - 1, 1)
category_number = to_categorical(category_number)

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(SEQ_LEN - 1, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(category_number.shape[1], activation='softmax'))

opt = Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

history = model.fit(
    nn_input_matrix, category_number, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.33
)

print(model.summary())

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
