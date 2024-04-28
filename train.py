import tensorflow as tf
import os
from argparse import ArgumentParser
import data
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--max-length", default=200, type=int)
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

    print('============================================')
    print('Training finished!')
    print('============================================')

    # transformer_encoder.save(args.path, save_format="tf")

    # Evaluate the model
    # test_loss, test_acc = transformer_encoder.evaluate(x_test, y_test)
    # print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
