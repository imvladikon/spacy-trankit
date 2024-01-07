#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0114, C0116, W0613
from typing import Optional

from spacy import blank, Language

from spacy_trankit import tokenizer
from spacy_trankit.version import __version__


def load(lang: str, name: Optional[str] = None, **kwargs) -> Language:
    config = {"nlp": {"tokenizer": {}}}
    if name is None:
        name = lang
    # fmt: off
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_trankit.PipelineAsTokenizer.v1"  # noqa: E501
    # fmt: on
    config["nlp"]["tokenizer"]["lang"] = lang
    for key, value in kwargs.items():
        config["nlp"]["tokenizer"][key] = value
    return blank(name, config=config)


def load_from_path(name: str, path: str, **kwargs) -> Language:
    config = {"nlp": {"tokenizer": {}}}
    # fmt: off
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_trankit.PipelineAsTokenizer.v1"  # noqa: E501
    # fmt: on
    config["nlp"]["tokenizer"]["cache_dir"] = path
    config["nlp"]["tokenizer"]["lang"] = name
    for key, value in kwargs.items():
        config["nlp"]["tokenizer"][key] = value
    return blank(name, config=config)
