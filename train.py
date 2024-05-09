import os
import tensorflow as tf
from argparse import ArgumentParser
import data
import trainer
import model


if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--max-length-input", default=30, type=int)
    parser.add_argument("--max-length-target", default=30, type=int)
    parser.add_argument("--num-heads-attention", default=2, type=int)
    parser.add_argument("--dff", default=128, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=32, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--dropout-rate", default=0.3, type=float)
    parser.add_argument("--path-train", required=True, type=str)
    parser.add_argument("--path-valid", required=True, type=str)
    parser.add_argument("--path-test", required=True, type=str)
    parser.add_argument("--checkpoint-path",
                        default='model_checkpoint', type=str)

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

    # Load data
    print('=============Data Processing Progress================')
    print('----------------Begin--------------------')
    print('Loading data ......')
    dataset = data.Data_Preprocessing(
        args.path_train, args.path_valid, args.path_test)

    print('Data processing ......')
    train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer = dataset.data_process(max_input_length=args.max_length_input,
                                                                                                       max_target_length=args.max_length_target,
                                                                                                       batch_size=args.batch_size)

    print('Suscessfully processed data !')
    print('----------------------------------------')
    # Training model

    transformer_model = model.Transformer(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads_attention, dff=args.dff, input_vocab_size=len(input_tokenizer.word_index),
                                          target_vocab_size=len(target_tokenizer.word_index), pe_input=args.max_length_input, pe_target=args.max_length_target, rate=args.dropout_rate)

    learning_rate = model.CustomSchedule(args.d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # , clipvalue=0.5

    trainer = trainer.Trainer(
        transformer_model, optimizer, args.epochs, args.checkpoint_path)

    trainer.fit(train_dataset, val_dataset, test_dataset)

    # transformer_encoder.save(args.path, save_format="tf")

    # Evaluate the model
    # trainer.evaluate(test_dataset, transformer_model)
