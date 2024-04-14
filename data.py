from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


def load_dataset(vocab_size):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    return (x_train, y_train), (x_test, y_test)


def get_preprocessed_data(vocab_size, max_length):
    (x_train, y_train), (x_test, y_test) = load_dataset(vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_length)
    x_test = pad_sequences(x_test, maxlen=max_length)
    return (x_train, y_train), (x_test, y_test)
