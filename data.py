import tensorflow as tf


def load_dataset(vocab_size):
    # số lượng từ cao nhất muốn giữ lại trong tập dữ liệu
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=vocab_size)
    return (x_train, y_train), (x_test, y_test)


def get_preprocessed_data(vocab_size, max_length):
    (x_train, y_train), (x_test, y_test) = load_dataset(vocab_size)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_length)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_length)
    return (x_train, y_train), (x_test, y_test)


def predict_data_preprocessing(text_list, max_length):
    for index in range(len(text_list)):
        text = text_list[index]
        word_to_index = tf.keras.datasets.imdb.get_word_index()
        text_indices = [word_to_index[word]
                        if word in word_to_index else 0 for word in text.split()]
        text_list[index] = text_indices

    text_list = tf.keras.preprocessing.sequence.pad_sequences(
        text_list, maxlen=max_length)
    return text_list
