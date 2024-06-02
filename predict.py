import tensorflow as tf
import model
from argparse import ArgumentParser
import data
import os
import trainer

if __name__ == '__main__':
    parser = ArgumentParser()

    # parser.add_argument("--max-length-input", default=64, type=int)
    # parser.add_argument("--max-length-target", default=64, type=int)
    # parser.add_argument("--num-heads-attention", default=1, type=int)
    # parser.add_argument("--vocab-size", default=20000, type=int)
    # parser.add_argument("--dff", default=2, type=int)
    # parser.add_argument("--num-layers", default=1, type=int)
    # parser.add_argument("--d-model", default=2, type=int)
    # parser.add_argument("--batch-size", default=512, type=int)
    # parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--dropout-rate", default=0.1, type=float)

    # Khai báo tham số cần thiết
    parser.add_argument("--max-length-input", default=64, type=int)
    parser.add_argument("--max-length-target", default=64, type=int)
    parser.add_argument("--num-heads-attention", default=1, type=int)
    parser.add_argument("--vocab-size", default=20000, type=int)
    parser.add_argument("--dff", default=2, type=int)
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--d-model", default=2, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--path-token-en", required=True, type=str)
    parser.add_argument("--path-token-vi", required=True, type=str)
    parser.add_argument("--model-path",
                        default='model.weights.h5', type=str)
    parser.add_argument("--predict-path", required=True, type=str)
    home_dir = os.getcwd()
    args = parser.parse_args()

    print('Data Processing Progress....')
    data_predict = data.Data_Predict(
        args.predict_path, args.path_token_en, args.path_token_vi)

    input_tokenizer, target_tokenizer = data_predict.get_tokenizer()

    # get encoder
    encoder_input = data_predict.predict_data_preprocessing(
        args.max_length_input)

    # get start and end token of decoder
    start, end = target_tokenizer.get_vocab(
    )['<s>'], target_tokenizer.get_vocab()['</s>']

    decoder_input = tf.convert_to_tensor(
        [[start]] * tf.shape(encoder_input)[0].numpy(), dtype=tf.int64)

    print('Suscessful data processing')

    # load model
    transformer_model = model.Transformer(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads_attention, dff=args.dff, input_vocab_size=len(input_tokenizer.get_vocab()),
                                          target_vocab_size=len(target_tokenizer.get_vocab()), pe_input=args.max_length_input, pe_target=args.max_length_target, rate=args.dropout_rate)
    learning_rate = model.CustomSchedule(args.d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # , clipvalue=0.5

    trainer_ = trainer.Trainer(
        transformer_model, optimizer, 0, args.model_path, start_token=start, end_token=end, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)

    print('=============Inference Progress================')
    print('----------------Begin--------------------')

    result = trainer_.predict(
        encoder_input, decoder_input, False, args.max_length_target)
    print(result)
