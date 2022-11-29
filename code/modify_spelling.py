# -*- coding: utf-8 -*-

import re

REPLACEMENTS = [
    (r"[\.\(\)\!\-@\,;:'\|]\"\<\>\?\*", r" "),
    (r"  +", r" "),
    (r"ph", r"f"),
    (r"sh", r"s"),
    (r"sch", r"s"),
    (r"k", r"c"),
    (r"ç", r"c"),
    (r"ß", r"s"),
    (r"[àáâãäåæ]", r"a"),
    (r"[èéêë]", r"e"),
    (r"[ìíîï]", r"i"),
    (r"[òóôõö]", r"o"),
    (r"[ùúûü]", r"u"),
    (r"y", r"i"),
    (r"ie", r"e"),
    (r"ei", r"e"),
    (r"ou", r"o"),
    (r"([qrtpsdfgjklxcvbnm])h", r"\1"),
    (r"([qrtpsdfgjklxcvbnm])\1+", r"\1"),
    (r"[rns]+($|\W)", r"\1"),
    (r"[aeiou]+($|\W)", r"\1")]


def modify_spelling(string):
    string = string.lower()
    for pattern, repl in REPLACEMENTS:
        string = re.sub(pattern, repl, string)
    return string


def modify_spelling_in_column(df, column_name, suffix = "_mod"):
    df[column_name + suffix] = df[column_name].fillna('').apply(modify_spelling)
    return df