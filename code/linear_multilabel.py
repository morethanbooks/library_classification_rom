
import pickle as pkl
import json
import numpy as np
import os
import pandas as pd

from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer

from hierarchy import map_to_max_level
from modify_spelling import modify_spelling
from utils import get_metrics


def featurize_column(column, threshold_length_features=None, **kwargs):
    vec = CountVectorizer(**kwargs).fit(column)
    features = vec.transform(column)
    names = vec.get_feature_names()
    if threshold_length_features is not None:
        index = [idx for idx, name in enumerate(names) if len(name) >= threshold_length_features]
        features, names = features[:, index], [names[idx] for idx in index]
    return features, names


def tokenize_column(column, tokenizer):
    return column.apply(lambda row: ' '.join(tokenizer.tokenize(row)))


def process_data(df, tokenizer):
    joined = [
        'title_supplement', 'summary',
        'title_continuing_resource', 'work_info',
        'work_title','expression_info',
        'expression_title', 
        'RVK_j', 
        'keyword', 'keyword_loc']

    joined_features, joined_names = featurize_column(
        tokenize_column(df[joined].fillna('').apply(' '.join, axis=1), tokenizer),
        # token_pattern=r"[^\| ]+", 
        max_features=5000)

    title_features, title_names = featurize_column(
        tokenize_column(df['title'].fillna(''), tokenizer),
        # token_pattern=r"[^\| ]+", 
        max_features=5000)

    publisher_features, publisher_names = featurize_column(
        tokenize_column(df['publisher'].fillna(''), tokenizer),
        # token_pattern=r"[^\| ]+",
        max_features=5000)

    language_features, language_names = featurize_column(
        tokenize_column(df['language_text'].fillna(''), tokenizer),
        max_features=1000)

    features = hstack(
        [joined_features, title_features, publisher_features, language_features])
    names = joined_names + title_names + publisher_names + language_names

    return features.tocsr(), names


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--dev-data', required=True)
    parser.add_argument('--max-level', type=int, default=0)
    parser.add_argument('--label-hierarchy', default='./data/label-hierarchy.tsv')
    parser.add_argument('--output-path', default='./linear_multilabel.preds')
    args = parser.parse_args()

    # class Object():
    #     pass
    # args = Object()
    # args.label_hierarchy = '../data/label-hierarchy.tsv'
    # args.train_data = '../data/1980_2019_data.train.parquet'
    # args.dev_data = '../data/1980_2019_data.dev.parquet'
    # args.max_level = 2

    hierarchy = pd.read_csv(args.label_hierarchy, sep='\t')
    if args.train_data.endswith('.parquet'):
        train = pd.read_parquet(args.train_data)
    else:
        train = pd.read_csv(args.train_data)
    if args.dev_data.endswith('.parquet'):
        dev = pd.read_parquet(args.dev_data)
    else:
        dev = pd.read_csv(args.dev_data)

    # map labels
    if args.max_level > 0:
        mapping = map_to_max_level(hierarchy, args.max_level)
        train['BK_split'] = train['BK_split'].apply(lambda row: tuple(set(map(mapping.get, row))))
        dev['BK_split'] = dev['BK_split'].apply(lambda row: tuple(set(map(mapping.get, row))))

    # filter languages
    target_languages = train['language_text'].value_counts()[
        list(train['language_text'].value_counts() > 1000)
    ].keys()
    target_languages = [lang for lang in target_languages if '|' not in lang]
    train = train[train['language_text'].isin(target_languages)]
    dev = dev[dev['language_text'].isin(target_languages)]

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    features, names = process_data(pd.concat([train, dev]), tokenizer)
    features = features.tocsr()

    train_X, dev_X = features[:len(train)], features[len(train):]
    train_y = train['BK_split'].tolist()
    dev_y = dev['BK_split'].tolist()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_y + dev_y)
    train_y, dev_y = y[:len(train)], y[len(train):]
    print("Training with {} labels".format(len(mlb.classes_)))
    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(train_X, train_y)

    # evaluate
    preds = clf.predict(dev_X)
    print(json.dumps(get_metrics(dev_y, preds)))
    with open(args.output_path, 'wb') as f:
        np.savez(f, preds=preds, dev_y=dev_y, index=dev.index)
    with open(args.output_path + '.label_mapping', 'wb') as f:
        pkl.dump(mlb, f)