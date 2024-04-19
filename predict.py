import tensorflow as tf
from argparse import ArgumentParser
from Transformer_Encoder.encoder import TransformerEncoderPack
import numpy as np
import data
import os

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--predict-text", default=10000, type=str)

    home_dir = os.getcwd()
    args = parser.parse_args()

    lines_predict = []
    with open(args.predict_text, 'r') as file:
        # Đọc từng dòng của file và lưu vào mảng
        for line in file:
            # Thêm dòng vào mảng sau khi loại bỏ các ký tự trắng thừa
            lines_predict.append(line.strip())

    lines_predict_preprocessed = data.predict_data_preprocessing(
        lines_predict, 200)
    model_pretrained = tf.keras.models.load_model('model')
    print(lines_predict_preprocessed)
    prediction = model_pretrained.predict(np.array(lines_predict_preprocessed))

    print(prediction)
