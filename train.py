import tensorflow as tf
import Transformer_Encoder.encoder as encoder
import os
from argparse import ArgumentParser
import data
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--vocab-size", default=10000, type=int)
    # parser.add_argument("--max-length-input", default=200, type=int)
    parser.add_argument("--embedding-dim", default=32, type=int)
    parser.add_argument("--num-heads-attention", default=2, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-encoder-layers", default=2, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--path", required=True, type=str)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Hiển thị thông tin training
    print('---------------------Welcome to ProtonX Transformer Encoder-------------------')
    print('Github: ktoan911')
    print('Email: khanhtoan.forwork@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transformer Classifier model with hyper-params:')
    print('===========================')

    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    # Load and preprocess the IMDB dataset
    max_length_input = 200
    (x_train, y_train), (x_test, y_test) = data.get_preprocessed_data(
        args.vocab_size, max_length_input)

    # # checkpoint
    # checkpoint_filepath = 'model_pretrained.keras'
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     monitor='val_binary_accuracy',  # Chọn metric bạn muốn theo dõi
    #     mode='max',
    #     save_best_only=True,
    #     verbose=1
    # )

    # Khởi tạo TransformerPack
    transformer_encoder = encoder.TransformerEncoderPack(
        num_encoder_layers=args.num_encoder_layers,
        d_model=args.d_model,
        num_heads=args.num_heads_attention,
        dff=args.dff,
        input_vocab_size=args.vocab_size,
        maximum_position_encoding=max_length_input,
        rate=args.dropout_rate)

    transformer_encoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Train the model
    transformer_encoder.fit(x_train, y_train, batch_size=args.batch_size,
                            epochs=args.epochs, validation_data=(x_test, y_test))

    print('============================================')
    print('Training finished!')
    print('============================================')

    transformer_encoder.save(args.path, save_format="tf")

    # Evaluate the model
    test_loss, test_acc = transformer_encoder.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
