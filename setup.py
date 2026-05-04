#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0114, C0116, W0613, C0103
import importlib
import importlib.util
import os

import setuptools

__package_name__ = "spacy_trankit"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def __import_file__(filename: str):
    spec = importlib.util.spec_from_file_location(os.path.basename(filename), filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_version(package_name: str):
    module = __import_file__(os.path.join(__location__, package_name, "version.py"))
    return module.__version__


__version__ = get_version(__package_name__)


def read_requirements():
    """Parses requirements from requirements.txt"""
    reqs_path = os.path.join(__location__, "requirements.txt")
    with open(reqs_path, encoding="utf8") as f:
        reqs = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return {"install_requires": reqs}


def read_readme():
    with open(os.path.join(__location__, "README.md"), "r", encoding="utf8") as f:
        long_description = f.read()
    return long_description


if __name__ == "__main__":
    setuptools.setup(
        name=__package_name__.replace("_", "-"),
        version=__version__,
        author="Vladimir Gurevich",
        description="spacy wrapper for Trankit, a Transformer-based multilingual "
        "neural dependency parser with tokenization and NER",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/imvladikon/spacy-trankit",  # noqa
        packages=setuptools.find_packages(
            exclude=(
                "tests",
                "tests.*",
            )
        ),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "Topic :: Text Processing :: Linguistic",
        ],
        python_requires=">=3.9",
        package_dir={__package_name__: __package_name__},
        **read_requirements()
    )
