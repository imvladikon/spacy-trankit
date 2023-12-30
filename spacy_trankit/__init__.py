#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from spacy_trankit.version import __version__
from typing import Optional
from spacy import blank, Language
from spacy_trankit import tokenizer


def load(lang: str, name: Optional[str] = None, **kwargs) -> Language:
    config = {"nlp": {"tokenizer": {}}}
    if name is None:
        name = lang
    config["nlp"]["tokenizer"][
        "@tokenizers"
    ] = "spacy_trankit.PipelineAsTokenizer.v1"  # noqa: E501
    config["nlp"]["tokenizer"]["lang"] = lang
    for key, value in kwargs.items():
        config["nlp"]["tokenizer"][key] = value
    return blank(name, config=config)


def load_from_path(name: str, path: str, **kwargs) -> Language:
    config = {"nlp": {"tokenizer": {}}}
    config["nlp"]["tokenizer"][
        "@tokenizers"
    ] = "spacy_trankit.PipelineAsTokenizer.v1"  # noqa: E501
    config["nlp"]["tokenizer"]["cache_dir"] = path
    config["nlp"]["tokenizer"]["lang"] = name
    for key, value in kwargs.items():
        config["nlp"]["tokenizer"][key] = value
    return blank(name, config=config)
