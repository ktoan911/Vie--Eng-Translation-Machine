import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
import data
import os

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--path-model", default='model_checkpoint', type=str)
    parser.add_argument("--path-tokenizer_vie", required=True, type=str)
    parser.add_argument("--path-tokenizer_en", required=True, type=str)
    parser.add_argument("--predict-data", required=True, type=str)
    parser.add_argument("--max-length", default=30, type=int)
    home_dir = os.getcwd()
    args = parser.parse_args()

    data_predict = data.Data_Predict(
        args.predict_data, args.path_tokenizer_en, args.path_tokenizer_vie)

    print(len(data_predict.tokenizer_vi.word_index))

    # input_tensor, tokenizer = data_predict.detokenizer(
    #     [[1, 21, 32, 43, 544, 65, 27]])

    # model_pretrained = tf.keras.models.load_model('args.path_model')
