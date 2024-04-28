import tensorflow as tf
from datasets import load_dataset


def load_datasets_default():
    # load_dataset
    dataset = load_dataset('mt_eng_vietnamese', 'iwslt2015-en-vi')

    # split dataset
    train_datatset = dataset['train']["translation"]
    val_datatset = dataset['validation']["translation"]
    test_datatset = dataset['test']["translation"]
