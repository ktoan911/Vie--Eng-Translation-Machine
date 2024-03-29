from keras.datasets import imdb
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalAveragePooling1D, Dense
from keras.models import Sequential
import Transformer_Encoder.encoder as encoder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':
    # Tham số
    vocab_size = 10000
    maxlen = 200
    embedding_dim = 32
    num_heads = 2
    d_model = 128
    dff = 512
    num_encoder_layers = 2
    batch_size = 32
    num_epochs = 10

    # Load and preprocess the IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    # Khởi tạo TransformerPack
    transformer = encoder.TransformerPack(
        num_encoder_layers=num_encoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        maximum_position_encoding=maxlen)

    transformer.compile(optimizer=Adam(), loss=BinaryCrossentropy(),
                        metrics=[BinaryAccuracy()])

    # Train the model
    transformer.fit(x_train, y_train, batch_size=batch_size,
                    epochs=3, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = transformer.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

    # In ra kiến trúc của mod
