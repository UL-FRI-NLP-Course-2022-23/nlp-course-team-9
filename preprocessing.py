import custom_dataset
import os
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split

def prepare_datasets(data_path, train_size=0.6, validation_size=0.2, test_size=0.2, random_seed=1):
    if train_size + validation_size + test_size != 1.0:
        raise Exception("Train, validation and test set percentages don't sum to 1.")

    df = pd.DataFrame(pd.read_pickle(data_path))

    train, validation_and_test = train_test_split(df, test_size=validation_size + test_size, shuffle=True, random_state=random_seed)
    validation, test = train_test_split(validation_and_test, test_size=validation_size/(validation_size + test_size))

    train = custom_dataset.MyDataSet(train)
    validation = custom_dataset.MyDataSet(validation)
    test = custom_dataset.MyDataSet(test)

    return train, validation, test


def merge_pkls(base_dir="data/"):
    all_paragraphs = []

    for f in os.listdir(base_dir):
        f_split = f.split('.')
        if len(f_split) == 2:
            filename, extension = f_split
            if extension == 'pkl':
                with open(base_dir + f, 'rb') as pkl_file:
                    pkl_paragraphs = pkl.load(pkl_file)
                    all_paragraphs += pkl_paragraphs

    with open(base_dir + '3rd_try.pkl', 'wb') as pkl_file:
        pkl.dump(all_paragraphs, pkl_file)


def split_datasets():
    train, validation, test = prepare_datasets("data/4th_try.pkl")
    pd.to_pickle(train, "data/4th_train.pkl")
    pd.to_pickle(validation, "data/4th_val.pkl")
    pd.to_pickle(test, "data/4th_test.pkl")


if __name__ == '__main__':
    split_datasets()
