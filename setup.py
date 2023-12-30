#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import os

import setuptools

__package_name__ = "spacy_trankit"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def __import_file__(filename: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(os.path.basename(filename), filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_version(package_name: str):
    module = __import_file__(os.path.join(__location__, package_name, "version.py"))
    return module.__version__


__version__ = get_version(__package_name__)


def get_extra_requires(path, add_all: bool = True) -> dict:
    import re
    from collections import defaultdict

    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps["all"] = set(vv for v in extra_deps.values() for vv in v)
    return extra_deps


def read_requirements():
    """Parses requirements from requirements.txt"""
    reqs_path = os.path.join(__location__, "requirements.txt")
    with open(reqs_path, encoding="utf8") as f:
        reqs = [line.strip() for line in f if not line.strip().startswith("#")]
    return {"install_requires": reqs}


def read_readme():
    with open(os.path.join(__location__, "README.md"), "r") as f:
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
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.8",
        package_dir={__package_name__: __package_name__},
        **read_requirements()
    )
