import os
import tensorflow as tf
from argparse import ArgumentParser
import data
import trainer
import model


if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--max-length-input", default=64, type=int)
    parser.add_argument("--max-length-target", default=64, type=int)
    parser.add_argument("--num-heads-attention", default=8, type=int)
    parser.add_argument("--vocab-size", default=20000, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--path-train", required=True, type=str)
    parser.add_argument("--path-valid", required=True, type=str)
    parser.add_argument("--path-test", required=True, type=str)
    parser.add_argument("--path-token-en", type=str)
    parser.add_argument("--path-token-vi", type=str)
    parser.add_argument("--model-path",
                        default='model.weights.h5', type=str)

    # parser.add_argument("--max-length-input", default=64, type=int)
    # parser.add_argument("--max-length-target", default=64, type=int)
    # parser.add_argument("--num-heads-attention", default=1, type=int)
    # parser.add_argument("--vocab-size", default=20000, type=int)
    # parser.add_argument("--dff", default=2, type=int)
    # parser.add_argument("--num-layers", default=1, type=int)
    # parser.add_argument("--d-model", default=2, type=int)
    # parser.add_argument("--batch-size", default=256, type=int)
    # parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--dropout-rate", default=0.1, type=float)
    # parser.add_argument("--path-train", required=True, type=str)
    # parser.add_argument("--path-valid", required=True, type=str)
    # parser.add_argument("--path-test", required=True, type=str)
    # parser.add_argument("--path-token-en", type=str)
    # parser.add_argument("--path-token-vi", type=str)
    # parser.add_argument("--model-path",
    #                     default='model.weights.h5', type=str)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Hiển thị thông tin training
    print('---------------------Welcome to ProtonX Translation Machine-------------------')
    print('Github: ktoan911')
    print('Email: khanhtoan.forwork@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transformer Classifier model with hyper-params:')
    print('===========================')

    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    # Load data
    print('=============Data Processing Progress================')
    print('----------------Begin--------------------')

    print('Loading data ......')

    user_input = input(
        "Do you want to use your data (y/n): ").strip().lower()
    if user_input != 'y':

        dataset = data.Data_Preprocessing(
            args.path_train, args.path_valid, args.path_test, vocab_size=args.vocab_size, type_data='arrow')

        print('Data processing ......')

        train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer = dataset.load_data_tokenizer(
            tokenizer_en_path=args.path_token_en,
            tokenizer_vi_path=args.path_token_vi,
            batch_size=args.batch_size, shuffle=True)

    else:
        path_save_train = r'Arrow_file\train.arrow'
        path_save_valid = r'Arrow_file\valid.arrow'
        path_save_test = r'Arrow_file\test.arrow'

        dataset = data.Data_Preprocessing(
            args.path_train, args.path_valid, args.path_test, vocab_size=args.vocab_size, type_data='csv')
        max_input_length = args.max_length_input
        max_target_length = args.max_length_target

        print('Data processing ......')

        train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer = dataset.data_process(
            max_input_length, max_target_length, path_save_train, path_save_test, path_save_valid)

        train_dataset = dataset.convert_to_tf_dataset(
            train_dataset, batch_size=args.batch_size)
        val_dataset = dataset.convert_to_tf_dataset(
            val_dataset, batch_size=args.batch_size)
        test_dataset = dataset.convert_to_tf_dataset(
            test_dataset, batch_size=args.batch_size)

    print('Suscessfully processed data !')
    print('----------------------------------------')
    # Training model

    transformer_model = model.Transformer(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads_attention, dff=args.dff, input_vocab_size=len(input_tokenizer.get_vocab()),
                                          target_vocab_size=len(target_tokenizer.get_vocab()), pe_input=args.max_length_input, pe_target=args.max_length_target, rate=args.dropout_rate)

    learning_rate = model.CustomSchedule(args.d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # , clipvalue=0.5

    trainer = trainer.Trainer(
        model=transformer_model, optimizer=optimizer, epochs=args.epochs, model_path=args.model_path, start_token=target_tokenizer.get_vocab()[
            '<s>'], end_token=target_tokenizer.get_vocab()['</s>'], tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)

    trainer.fit(train_dataset, val_dataset, test_dataset)

    print('----------------------------------------')
    print('Training model successfully !')
