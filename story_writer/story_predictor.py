from text_processor import process_test_text, create_series_of_words, form_input_matrix
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

SEQ_LEN = 60
tokenizer = Tokenizer()
tokenized_text, number_to_word = process_test_text(0.00, 0.01)
tokenized_text = create_series_of_words(tokenized_text, SEQ_LEN)
nn_input_matrix, category_number = form_input_matrix(tokenized_text)

predicted_classes = loaded_model.predict_classes(nn_input_matrix)

word_dict = dict((v, k) for k, v in number_to_word.items())
print(word_dict[predicted_classes[0]])

"""take in 60 words, predict new word, append this new word, predict with these 60 new words and repeat"""

first_59_words = nn_input_matrix[0]
print(first_59_words)
predicted_paragraph =[]
for i in range(100):
    new_word = loaded_model.predict_classes(first_59_words[i: i + 59])
    first_59_words.append(word_dict[new_word])
print(first_59_words)
