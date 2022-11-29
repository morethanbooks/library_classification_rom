# %%
import sys
import os
#import glob
#import re 
import pandas as pd
import numpy as np
from lxml import etree
#import requests
#import time
from collections import Counter
#import shutil
from transformers import BertTokenizer #, AutoTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

print("all good")





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



# %%
#sys.path.append(os.path.abspath("C:/Users/calvotello/Dropbox/MTB/Göttingen/research/"))
#sys.path.append(os.path.abspath("./../../"))


# %%
#import tokenize #from librarian_robot import tokenize



# %%

df = pd.read_parquet("./../../library_classification_rom_priv_data/1980_2019_data_labels_tokenized.parquet")

print("read data")


# %%
df

# %% [markdown]
# # Extract Tokens

# %%
#tokenize.tokenize_column?

# %%
tokens_titles_df = tokenize_column(df, "title_tokenized", reg_ex = r"[^\| ]+", max_features = 5000)

print("extracted tokens from titles")

# %%
tokens_titles_df

# %%
tokens_titles_df.to_parquet("./../../library_classification_rom_priv_data/tokens_title_1980_2019_5000.parquet")
print("saved")
# %%


# %%


# %%
tokens_joined_column_df = tokenize_column(df, "joined_column_tokenized", reg_ex = r"[^\| ]+", max_features = 5000)

print("extracted tokens from joined_column_tokenized")

# %%
tokens_joined_column_df

# %%
tokens_joined_column_df.to_parquet("./../../library_classification_rom_priv_data/tokens_joined_column_1980_2019_5000.parquet")
print("saved")

# %%
tokens_joined_column_df

# %%
tokens_publisher_df = tokenize_column(df, "publisher_tokenized", reg_ex = r"[^\| ]+", max_features = 5000)

print("extracted tokens from publisher_tokenized")
# %%
tokens_publisher_df

# %%
tokens_publisher_df.to_parquet("./../../library_classification_rom_priv_data/tokens_publisher_balanced_1980_2019_5000.parquet")
print("saved")

# %%
#df = pd.read_parquet("./../data/2020_data_balanced_with_labels.parquet")




# %%
df

# %%
df[["language_text"]]

# %%
languages_df = tokenize_column(df, "language_text",  ngram_range=(1,1), analyzer="word",  reg_ex='(?u)\\b\\w+\\b', max_features = 1000, add_name_of_original_column = True)

print("extracted tokens from language_text")
# %%
languages_df

# %%
#languages_df.sum().iloc[0:30].plot.bar()

# %%
df = pd.merge(df, languages_df, left_index= True, right_index= True)

# %%
df

# %%

df.to_parquet("./../../library_classification_rom_priv_data/1980_2019_data_labels_tokenized_languages.parquet")
print("saved")


print("merging")
# %%
tokens_joined_df = tokens_titles_df

# %%

tokens_joined_df = pd.merge(tokens_joined_df, languages_df, right_index=True, left_index=True)

# %%
tokens_joined_df

# %%

tokens_joined_df = pd.merge(tokens_joined_df, tokens_joined_column_df, right_index=True, left_index=True)

# %%
del tokens_joined_column_df

# %%
del tokens_titles_df

# %%
del df

# %%

tokens_joined_df = pd.merge(tokens_joined_df, tokens_publisher_df, right_index=True, left_index=True)

# %%
tokens_joined_df

# %%
del tokens_publisher_df

# %%
del languages_df

# %%
tokens_joined_df.to_parquet("./../../library_classification_rom_priv_data/tokens_joined_1980_2019.parquet")

print("saved merged")
# %%
#tokens_joined_df = pd.read_parquet("./../data/tokens_joined_2005_2019.parquet")

# %%
tokens_joined_df = tokens_joined_df.astype(bool).astype(int)

# %%
tokens_joined_df.to_parquet("./../../library_classification_rom_priv_data/tokens_joined_binary_1980_2019.parquet")
print("saved binary")


tokens_joined_df.sample(5000, random_state = 2022).to_parquet("./../../library_classification_rom_priv_data/tokens_joined_binary_1980_2019_sample_5000.parquet")
print("finished")