import tensorflow as tf
import pandas as pd
from datasets import load_dataset
import underthesea
import pickle
import re


class Data_Preprocessing:
    def __init__(self, train_path, val_path, test_path):
        self.train_dataset = pd.read_csv(train_path, encoding='utf-8')
        self.val_dataset = pd.read_csv(val_path, encoding='utf-8')
        self.test_dataset = pd.read_csv(test_path, encoding='utf-8')

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

    def preprocess_sentence(self, sentence):
        sentence = str(sentence).replace("_", " ")
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<start>', sentence, '<end>')
        return sentence

    def split_envi(self, examples):
        examples['en'] = examples['en'].map(self.preprocess_sentence)
        examples['vi'] = examples['vi'].map(self.preprocess_sentence)
        inputs = [ex for ex in examples['en']]
        targets = [ex for ex in examples['vi']]
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
        dataset = dataset.shuffle(10000).batch(
            batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def split_input_target(self, en, vi):
        input_en = tf.convert_to_tensor(en, dtype=tf.int64)
        input_vi = tf.convert_to_tensor(vi[:, :-1], dtype=tf.int64)
        target_vi = tf.convert_to_tensor(vi[:, 1:], dtype=tf.int64)
        return (input_en, input_vi), target_vi

    def data_process(self, max_input_length, max_target_length, batch_size=32):
        train_dataset_inptensor, train_datasetout_tensor, input_tokenizer, target_tokenizer = self.preprocess(
            self.train_dataset, max_input_length, max_target_length)
        val_dataset_inptensor, val_datasetout_tensor, _, _ = self.preprocess(self.val_dataset,
                                                                             max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)
        test_dataset_inptensor, test_datasetout_tensor, _, _ = self.preprocess(self.test_dataset,
                                                                               max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)

        train_dataset_inptensor, train_datasetout_tensor = self.split_input_target(
            train_dataset_inptensor, train_datasetout_tensor)
        val_dataset_inptensor, val_datasetout_tensor = self.split_input_target(val_dataset_inptensor,
                                                                               val_datasetout_tensor)
        test_dataset_inptensor, test_datasetout_tensor = self.split_input_target(test_dataset_inptensor,
                                                                                 test_datasetout_tensor)

        with open(r'Tokenizer\en_tokenizer.pkl', 'wb') as handle:
            pickle.dump(input_tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(r'Tokenizer\vi_tokenizer.pkl', 'wb') as handle:
            pickle.dump(target_tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        train_dataset = self.convert_tfdataset(
            train_dataset_inptensor, train_datasetout_tensor, batch_size)
        val_dataset = self.convert_tfdataset(val_dataset_inptensor,
                                             val_datasetout_tensor, batch_size)
        test_dataset = self.convert_tfdataset(test_dataset_inptensor,
                                              test_datasetout_tensor, batch_size)
        return train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer


class Data_Predict:
    def __init__(self, data_path, tokenizer_en_path, tokenizer_vi_path):
        self.data = self.load_data(data_path)
        self.tokenizer_en, self.tokenizer_vi = self.load_tokenizer(
            tokenizer_en_path, tokenizer_vi_path)

    def load_tokenizer(self, tokenizer_en_path, tokenizer_vi_path):
        with open(tokenizer_en_path, 'rb') as handle:
            tokenizer_en = pickle.load(handle)

        with open(tokenizer_vi_path, 'rb') as handle:
            tokenizer_vi = pickle.load(handle)
        return tokenizer_en, tokenizer_vi

    def load_data(self, data_path):
        lines_predict = []
        with open(data_path, 'r') as file:
            # Đọc từng dòng của file và lưu vào mảng
            for line in file:
                # Thêm dòng vào mảng sau khi loại bỏ các ký tự trắng thừa
                lines_predict.append(line.strip())
        return lines_predict

    def preprocess_sentence(self, sentence):
        sentence = str(sentence)
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<start>', sentence, '<end>')
        return sentence

    def predict_data_preprocessing(self, max_length):
        lines_predict_preprocessed = [self.preprocess_sentence(
            sentence) for sentence in self.data]
        input_en = self.tokenizer_en.texts_to_sequences(
            lines_predict_preprocessed)
        input_en = tf.keras.utils.pad_sequences(
            input_en, padding='post', maxlen=max_length)
        input_tensor = tf.convert_to_tensor(input_en, dtype=tf.int64)
        return input_tensor, self.tokenizer_en

    def detokenizer(self, tensor):
        detokenized_texts = self.tokenizer_vi.sequences_to_texts(tensor)
        return [sentence.replace('<start>', '').replace('<end>', '').replace('<unk>', '').replace('_', ' ') for sentence in detokenized_texts], self.tokenizer_vi
