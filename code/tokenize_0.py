# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:15:17 2017

@author: jose 


"""
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer #, AutoTokenizer



"""
def annotate_with_bert(df, columns_to_tokenize = ["title", "joint_column", "publisher"]):
    bert_multilingual_uncased_tz = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    for index, row in df.iterrows():
        for column in columns_to_tokenize:
            df[column] = df[column].fillna("").astype(str)
            df.loc[index, column + "_tokenized"] = "|".join(bert_multilingual_uncased_tz.tokenize(row[column]))
            df[column + "_tokenized"] = df[column + "_tokenized"].fillna("").astype(str)
    return df
    




#def extract_features_by_bert(df, columns_to_extract = ):

#    tokens_titles_df = tokenize_column(df, "title_tokenized", reg_ex = r"[^\| ]+", max_features = 5000)





def tokenize_entire_table(df,
        reg_ex = r"(?u)\b[\w_@]+\b|[¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡\"\'·+\{\#\[;­,«~]", 
        max_features = 5000,
    ):
    i = 0
    for column in df.columns.tolist():
        print(column)
        tokens_partial_df = tokenize_column(df, column, reg_ex = reg_ex, max_features = max_features)
        
        if i == 0:
            tokens_df = tokens_partial_df
        else:
            tokens_df = pd.merge(tokens_df, tokens_partial_df, left_index = True, right_index = True)
        i += 1


    tokens_df = tokens_df[tokens_df.sum().sort_values(ascending=False).index]
    tokens_df
    return tokens_df
"""

def tokenize_column(
    df,
    column_name,
    add_name_of_original_column = True,
    reg_ex = r"(?u)\b[\w_@]+\b|[¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡\"\'·+\{\#\[;­,«~]", # 
    sort_columns_by_frequency = True,
    lowercase = True,
    max_features = 100,
    analyzer = "word", #{‘word’, ‘char’, ‘char_wb’},
    ngram_range = (1,1), #ngram_range = None,
    binary = False,
    strip_columns = False,
    delete_repited_columns = False,
    threshold_length_features = None,
    vocabulary = None,
    ):
    """
    Regex for BK classes: reg_ex = r"(?u)\b\d\d\.\d\d\b"
    Regex for BERT tokens: reg_ex = [^\| ]+

    """


    vec = CountVectorizer(
            token_pattern = reg_ex,
            lowercase = lowercase,
            max_features = max_features,
            analyzer = analyzer,
            ngram_range = ngram_range,
            binary = binary,
            vocabulary = vocabulary
            )

    tokens = vec.fit_transform(df[column_name].astype(str).tolist())

    tokens_df = pd.DataFrame(tokens.toarray(), columns = vec.get_feature_names(), index = df.index.tolist())

    tokens_df.fillna(0, inplace=True)

    print("Shape of tokens: ", tokens_df.shape)

    if strip_columns == True:
        tokens_df.columns = [column_name_feature.strip()  for column_name_feature in tokens_df.columns.tolist()]

    if delete_repited_columns == True:
        tokens_df = tokens_df.sum(axis=1, level=0, skipna=False)


    if threshold_length_features not in [None, False]:
        try:
            tokens_df = tokens_df[[column for column in tokens_df.columns.tolist() if len(column) >= threshold_length_features]]
        except:
            print("error")


    if add_name_of_original_column == True:
        tokens_df.columns = [str(column_name_feature) + "@" + str(column_name)  for column_name_feature in tokens_df.columns.tolist()]

    if sort_columns_by_frequency == True:
        tokens_df = tokens_df[tokens_df.sum().sort_values(ascending=False).index]

    print("Shape of tokens after filtering: ", tokens_df.shape)

    return tokens_df

"""
contains_accents
contains_arabic
contains_hebrew
contains_chinese
contains_japanese
contains_cirilic
contains_latin
contains_latin_extended
contains_numbers
number_of_numbers
length_title
"""