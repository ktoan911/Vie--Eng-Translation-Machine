import data
import tensorflow as tf

if __name__ == '__main__':
    train_path = r"D:\Python\Translation-Machine\Data\train.csv"
    valid_path = r"D:\Python\Translation-Machine\Data\valid.csv"
    test_path = r"D:\Python\Translation-Machine\Data\test.csv"

    path_save_train = r'D:\Python\Translation-Machine\Arrow_file\train.arrow'
    path_save_valid = r'D:\Python\Translation-Machine\Arrow_file\valid.arrow'
    path_save_test = r'D:\Python\Translation-Machine\Arrow_file\test.arrow'

    dataset = data.Data_Preprocessing(
        train_path, valid_path, test_path, type_data='none')
    max_input_length = 30
    max_target_length = 30

    # train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer = dataset.data_process(
    #     max_input_length, max_target_length, path_save_train, path_save_test, path_save_valid)

    # print(train_dataset)

    dat = dataset.load_dataset(path_save_train)
    print(dat)
