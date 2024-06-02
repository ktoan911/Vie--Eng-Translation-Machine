import tensorflow as tf
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
import re
import os
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class Data_Preprocessing:
    def __init__(self, train_path=None, val_path=None, test_path=None, vocab_size=20000, type_data='csv'):
        if (train_path is None or val_path is None or test_path is None):
            return
        self.vocab_size = vocab_size
        if type_data == 'csv':
            self.train_dataset = pd.read_csv(train_path)
            self.val_dataset = pd.read_csv(val_path)
            self.test_dataset = pd.read_csv(test_path)
        if type_data == 'arrow':
            self.train_dataset = self.load_dataset(train_path)
            self.val_dataset = self.load_dataset(val_path)
            self.test_dataset = self.load_dataset(test_path)

    def train_tokenizer(self, dataset):
        # Create a tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Customize the pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Initialize trainer
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=[
                                      "<unk>", "<pad>", "<s>", "</s>"])

        # Train the tokenizer
        tokenizer.train_from_iterator(dataset, trainer)

        return tokenizer

    def tokenizer(self, dataset, language='en', tokenizer_en=None, tokenizer_vi=None):
        if language == "vi":
            # dataset = [underthesea.word_tokenize(
            #     text, format='text') for text in dataset]
            tokenizer = tokenizer_vi
        else:
            tokenizer = tokenizer_en

        if tokenizer is None:
            tokenizer = self.train_tokenizer(dataset)

        tokenized_dataset = [tokenizer.encode(text).ids for text in dataset]
        return tokenized_dataset, tokenizer

    def preprocess_sentence(self, sentence):
        sentence = str(sentence).replace("_", " ")
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<s>', sentence, '</s>')
        return sentence

    def split_envi(self, examples):
        examples['en'] = examples['en'].map(self.preprocess_sentence)
        examples['vi'] = examples['vi'].map(self.preprocess_sentence)
        inputs = [ex for ex in examples['en']]
        targets = [ex for ex in examples['vi']]
        return inputs, targets

    def padding(self, tensor, max_length):
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, padding='post', maxlen=max_length, truncating='post')
        return tensor

    def vi_process(self, text_list):
        tar_inp = []
        tar_out = []
        for text in text_list:
            tar_inp.append(text[:-1])
            tar_out.append(text[1:])
        return tar_inp, tar_out

    def preprocess(self, examples, max_input_length, max_target_length, tokenizer_en=None, tokenizer_vi=None):
        inputs, targets = self.split_envi(examples)
        input_tensor, input_tokenizer = self.tokenizer(
            dataset=inputs, language="en", tokenizer_en=tokenizer_en)
        target_tensor, target_tokenizer = self.tokenizer(
            dataset=targets, language='vi', tokenizer_vi=tokenizer_vi)

        tar_inp, tar_out = self.vi_process(target_tensor)

        input_tensor = self.padding(
            input_tensor, max_length=max_input_length)
        tar_inp = self.padding(
            tar_inp, max_length=max_target_length)
        tar_out = self.padding(
            tar_out, max_length=max_target_length)

        return input_tensor, tar_inp, tar_out, input_tokenizer, target_tokenizer

    def split_input_target(self, en, vi_in, vi_tar, path_save):
        input_en = []
        input_vi = []
        target_vi = []

        # Tạo các danh sách riêng biệt cho từng cột
        for f1, f2, label in zip(en, vi_in, vi_tar):
            input_en.append(f1)
            input_vi.append(f2)
            target_vi.append(label)

        # Tạo từ điển dữ liệu trực tiếp từ các danh sách đã tạo
        reformatted_data_dict = {
            'input_en': input_en,
            'input_vi': input_vi,
            'target_vi': target_vi
        }

        # Tạo Dataset từ từ điển
        hf_dataset = Dataset.from_dict(reformatted_data_dict)
        hf_dataset.save_to_disk(path_save)
        return hf_dataset

    def data_process(self, max_input_length, max_target_length, path_train, path_test, path_valid):
        train_dataset_inptensor, train_datasetout_tensor_inp, train_datasetout_tensor_out, input_tokenizer, target_tokenizer = self.preprocess(
            self.train_dataset, max_input_length, max_target_length)
        val_dataset_inptensor, val_datasetout_tensor_inp, val_datasetout_tensor_out, _, _ = self.preprocess(self.val_dataset,
                                                                                                            max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)
        test_dataset_inptensor, test_datasetout_tensor_inp, test_datasetout_tensor_out, _, _ = self.preprocess(self.test_dataset,
                                                                                                               max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)

        train_dataset = self.split_input_target(
            train_dataset_inptensor, train_datasetout_tensor_inp, train_datasetout_tensor_out, path_train)
        val_dataset = self.split_input_target(
            val_dataset_inptensor, val_datasetout_tensor_inp, val_datasetout_tensor_out, path_valid)
        test_dataset = self.split_input_target(
            test_dataset_inptensor, test_datasetout_tensor_inp, test_datasetout_tensor_out, path_test)

        input_tokenizer.save(f'Tokenizer/en_tokenizer.json')
        target_tokenizer.save(f'Tokenizer/vi_tokenizer.json')

        return train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer

    def load_dataset(self, path_load):
        if not os.path.exists(path_load):
            print(f"File {path_load} does not exist.")
            sys.exit(1)  # Kết thúc chương trình với mã lỗi 1
        else:
            print(f"File {path_load} exists.")
        return load_from_disk(path_load)

    def load_tokenizer(self, tokenizer_en_path, tokenizer_vi_path):
        if not os.path.exists(tokenizer_en_path):
            print(f"File {tokenizer_en_path} does not exist.")
            sys.exit(1)

        if not os.path.exists(tokenizer_vi_path):
            print(f"File {tokenizer_vi_path} does not exist.")
            sys.exit(1)

        tokenizer_en = Tokenizer.from_file(tokenizer_en_path)
        tokenizer_vi = Tokenizer.from_file(tokenizer_vi_path)

        return tokenizer_en, tokenizer_vi

    def convert_to_tf_dataset(self, hf_dataset, batch_size=32, shuffle=True):
        def encode(examples):
            # Handle NaN values
            examples['input_en'] = np.nan_to_num(examples['input_en'], nan=0.0)
            examples['input_vi'] = np.nan_to_num(examples['input_vi'], nan=0.0)
            examples['target_vi'] = np.nan_to_num(
                examples['target_vi'], nan=0.0)

            return {
                'input_en': tf.constant(examples['input_en']),
                'input_vi': tf.constant(examples['input_vi']),
                'target_vi': tf.constant(examples['target_vi'])
            }

        # Map the dataset to encode each example
        tf_dataset = hf_dataset.map(encode)

        # Convert to a tf.data.Dataset
        tf_dataset = tf_dataset.to_tf_dataset(
            columns=['input_en', 'input_vi'],
            label_cols='target_vi',
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=None  # Use the default collate function
        )

        return tf_dataset

    def load_data_tokenizer(self, tokenizer_en_path, tokenizer_vi_path, batch_size=32, shuffle=True):
        tokenizer_en, tokenizer_vi = self.load_tokenizer(
            tokenizer_en_path, tokenizer_vi_path)
        return self.convert_to_tf_dataset(self.train_dataset, batch_size, shuffle), self.convert_to_tf_dataset(self.val_dataset, batch_size, shuffle), self.convert_to_tf_dataset(self.test_dataset, batch_size, shuffle), tokenizer_en, tokenizer_vi


class Data_Predict:
    def __init__(self, data_path, tokenizer_en_path, tokenizer_vi_path):
        self.data = self.load_data(data_path)
        self.tokenizer_en, self.tokenizer_vi = self.load_tokenizer(
            tokenizer_en_path, tokenizer_vi_path)

    def get_tokenizer(self):
        return self.tokenizer_en, self.tokenizer_vi

    def load_tokenizer(self, tokenizer_en_path, tokenizer_vi_path):
        if not os.path.exists(tokenizer_en_path):
            print(f"File {tokenizer_en_path} does not exist.")
            sys.exit(1)

        if not os.path.exists(tokenizer_vi_path):
            print(f"File {tokenizer_vi_path} does not exist.")
            sys.exit(1)

        tokenizer_en = Tokenizer.from_file(tokenizer_en_path)
        tokenizer_vi = Tokenizer.from_file(tokenizer_vi_path)

        return tokenizer_en, tokenizer_vi

    def load_data(self, data_path, type_data='txt'):
        lines_predict = []
        if (type_data == 'txt'):
            with open(data_path, 'r') as file:
                # Đọc từng dòng của file và lưu vào mảng
                for line in file:
                    # Thêm dòng vào mảng sau khi loại bỏ các ký tự trắng thừa
                    lines_predict.append(line)
        return lines_predict

    def preprocess_sentence(self, sentence):
        sentence = str(sentence).replace("_", " ")
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<s>', sentence, '</s>')
        return sentence

    def predict_data_preprocessing(self, max_length):
        lines_predict_preprocessed = [self.preprocess_sentence(
            sentence) for sentence in self.data]

        input_en = [self.tokenizer_en.encode(
            text).ids for text in lines_predict_preprocessed]

        input_en = tf.keras.preprocessing.sequence.pad_sequences(
            input_en, padding='post', maxlen=max_length, truncating='post')
        input_tensor = tf.convert_to_tensor(input_en, dtype=tf.int64)
        return input_tensor
