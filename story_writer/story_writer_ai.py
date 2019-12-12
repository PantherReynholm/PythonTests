from keras.layers import Dropout, Dense, CuDNNLSTM, BatchNormalization, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from text_processor import process_training_text, create_series_of_words, form_input_matrix
from keras.optimizers import Adam
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

SEQ_LEN = 61
EPOCHS = 50
BATCH_SIZE = 64

tokenized_text = process_training_text(0.10)
tokenized_text = create_series_of_words(tokenized_text, SEQ_LEN)
np.random.shuffle(tokenized_text)
nn_input_matrix, category_number = form_input_matrix(tokenized_text)


model = Sequential()

model.add(CuDNNLSTM(128, input_shape=nn_input_matrix.shape[1:], return_sequences=True))
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
    loss='categorical_crossentropy',
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
