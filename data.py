import tensorflow as tf
import pandas as pd
from datasets import load_dataset
import underthesea


class Data:
    def __init__(self, train_path, val_path, test_path):
        self.train_dataset = pd.read_csv(train_path, encoding='utf-8')
        self.val_datatset = pd.read_csv(val_path, encoding='utf-8')
        self.test_datatset = pd.read_csv(test_path, encoding='utf-8')

    # split dataset

    def tokenizer(self, dataset, language='en', tokenizer_en=None, tokenizer_vi=None):
        if language == "vi":
            dataset = [underthesea.word_tokenize(
                text, format='text') for text in dataset]
            tokenizer = tokenizer_vi
        else:
            tokenizer = tokenizer_en
        if tokenizer is None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(
                filters='', oov_token='<unk>')
            tokenizer.fit_on_texts(dataset)
        tensor = tokenizer.texts_to_sequences(dataset)
        return tensor, tokenizer

    def split_envi(self, examples):
        source_lang = "en"
        target_lang = "vi"
        inputs = [str(ex) for ex in examples['en']]
        targets = [str(ex) for ex in examples['vi']]
        return inputs, targets

    def padding(self, tensor, max_length):
        tensor = tf.keras.utils.pad_sequences(
            tensor, padding='post', maxlen=max_length)
        return tensor

    def preprocess(self, examples, max_input_length, max_target_length, tokenizer_en=None, tokenizer_vi=None):
        inputs, targets = self.split_envi(examples)
        input_tensor, input_tokenizer = self.tokenizer(
            dataset=inputs, language="en", tokenizer_en=tokenizer_en)
        target_tensor, target_tokenizer = self.tokenizer(
            dataset=targets, language='vi', tokenizer_vi=tokenizer_vi)
        input_tensor = self.padding(
            input_tensor, max_length=max_input_length)
        target_tensor = self.padding(
            target_tensor, max_length=max_target_length)
        return input_tensor, target_tensor, input_tokenizer, target_tokenizer

    def convert_tfdataset(self, input_tensor, target_tensor, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor, target_tensor))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(1)
        return dataset

    def data_process(self, max_input_length, max_target_length, batch_size=32):
        train_dataset_inptensor, train_datasetout_tensor, input_tokenizer, target_tokenizer = self.preprocess(
            self.train_dataset, max_input_length, max_target_length)
        val_dataset_inptensor, val_datasetout_tensor, _, _ = self.preprocess(self.val_datatset,
                                                                             max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)
        test_dataset_inptensor, test_datasetout_tensor, _, _ = self.preprocess(self.test_datatset,
                                                                               max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)
        train_dataset = self.convert_tfdataset(
            train_dataset_inptensor, train_datasetout_tensor, batch_size)
        val_dataset = self.convert_tfdataset(val_dataset_inptensor,
                                             val_datasetout_tensor, batch_size)
        test_dataset = self.convert_tfdataset(test_dataset_inptensor,
                                              test_datasetout_tensor, batch_size)
        return train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer
