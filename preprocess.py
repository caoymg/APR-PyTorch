import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse

import numpy as np
import pandas as pd


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.train_fpath = os.path.join(data_dir, 'ml-1m.train.rating')
        self.test_fpath = os.path.join(data_dir, 'ml-1m.test.rating')

    def load(self):
        # Load data
        train_df = pd.read_csv(self.train_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        test_df = pd.read_csv(self.test_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return train_df, test_df


class Yelp(DatasetLoader):
    def __init__(self, data_dir):
        self.train_fpath = os.path.join(data_dir, 'yelp.train.rating')
        self.test_fpath = os.path.join(data_dir, 'yelp.test.rating')

    def load(self):
        # Load data
        train_df = pd.read_csv(self.train_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        test_df = pd.read_csv(self.test_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return train_df, test_df

class Pinterest20(DatasetLoader):
    def __init__(self, data_dir):
        self.train_fpath = os.path.join(data_dir, 'yelp.train.rating')
        self.test_fpath = os.path.join(data_dir, 'yelp.test.rating')

    def load(self):
        # Load data
        train_df = pd.read_csv(self.train_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        test_df = pd.read_csv(self.test_fpath,
                         sep='	',
                         engine='python',
                         names=['user', 'item', 'rate', 'time']).reset_index(drop=True)
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return train_df, test_df

def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user].append((row.time, row.item))
    return user_list


def split_train_test(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items
    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df)*test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)
    else:
        total_user_list = create_user_list(df, user_size)
        train_user_list = [None] * len(user_list)
        test_user_list = [None] * len(user_list)
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list)*(1-test_size)):]
            train_item = item_list[:math.ceil(len(item_list)*(1-test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
    # Remove time
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    return train_user_list, test_user_list


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def main(args):
    if args.dataset == 'ml-1m':
        train_df, test_df = MovieLens1M(args.data_dir).load()
    elif args.dataset == 'yelp':
        train_df, test_df = Yelp(args.data_dir).load()
    elif args.dataset == 'pinterest-20':
        train_df, test_df = Pinterest20(args.data_dir).load()
    else:
        raise NotImplementedError

    all_df = pd.concat([train_df, test_df]).drop_duplicates()
    train_user_size = len(train_df['user'].unique())
    test_user_size = len(test_df['user'].unique())
    user_size = len(all_df['user'].unique())
    item_size = len(all_df['item'].unique())
    train_user_list = create_user_list(train_df, train_user_size)
    test_user_list = create_user_list(test_df, test_user_size)
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    print('Complete spliting items for training and testing')

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair}
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(args.output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['ml-1m', 'yelp', 'pinterest-20'])
    parser.add_argument('--data_dir',
                        type=str,
                        # default=os.path.join('Data', 'ml-1m.'),==>'Data/ml-1m./train.rating'
                        default=os.path.join('Data'),
                        help="File path for raw data")
    parser.add_argument('--output_data',
                        type=str,
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
                        help="File path for preprocessed data")
    parser.add_argument('--time_order',
                        action='store_true',
                        help="Proportion for training and testing split")
    args = parser.parse_args()
    main(args)
