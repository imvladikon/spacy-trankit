<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + Trankit

This package wraps the [Trankit](https://github.com/nlp-uoregon/trankit) library, so you can use trankit models in a
[spaCy](https://spacy.io) pipeline. 


[![CI](https://github.com/imvladikon/spacy-trankit/actions/workflows/ci.yml/badge.svg)](https://github.com/imvladikon/spacy-trankit/actions/workflows/ci.yml)
[![PyPi](https://img.shields.io/pypi/v/spacy-trankit.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-trankit)
[![GitHub](https://img.shields.io/github/release/imvladikon/spacy-trankit/all.svg?style=flat-square)](https://github.com/imvladikon/spacy-trankit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

Using this wrapper, you'll be able to use the following annotations, computed by
your pretrained `trankit` pipeline/model:

- Statistical tokenization (reflected in the `Doc` and its tokens)
- Lemmatization (`token.lemma` and `token.lemma_`)
- Part-of-speech tagging (`token.tag`, `token.tag_`, `token.pos`, `token.pos_`)
- Morphological analysis (`token.morph`)
- Dependency parsing (`token.dep`, `token.dep_`, `token.head`)
- Named entity recognition (`doc.ents`, `token.ent_type`, `token.ent_type_`,
  `token.ent_iob`, `token.ent_iob_`)
- Sentence segmentation (`doc.sents`)
- Multiword token preservation for languages such as Arabic and Hebrew via
  `token._.trankit_expanded`

## ️️️⌛️ Installation

As of v0.2.0 `spacy-trankit` is only compatible with **spaCy v3.x**. To install
the most recent version:

```bash
pip install git+https://github.com/imvladikon/spacy-trankit
```

or from pypi:

```bash
pip install spacy-trankit
```


## 📖 Usage & Examples

Load pre-trained `trankit` model into a spaCy pipeline:

```python
import spacy_trankit

# Initialize the pipeline
nlp = spacy_trankit.load("en")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```

By default, `mwt_strategy="auto"` expands multiword tokens when the expanded
tokens can be aligned back to the original text without changing `doc.text`.
Expansions that cannot be represented as substrings of the original text are
kept non-destructive. For example, Arabic and Hebrew clitic expansions can
differ from the surface token, so the spaCy token keeps the original surface
form and stores Trankit's expansion under `token._.trankit_expanded`.

```python
doc = nlp("ذهبت للبيت اليوم")
for token in doc:
    print(token.text, token._.trankit_expanded)
```

If you always want surface tokens, pass `mwt_strategy="preserve"`. If you need
the previous expanded-token behavior and accept that spaCy may have to replace
the original text with space-separated expanded tokens for unalignable cases,
pass `mwt_strategy="expand"`:

```python
nlp = spacy_trankit.load("ar", mwt_strategy="preserve")
nlp = spacy_trankit.load("ar", mwt_strategy="expand")
```

Load it from the path:

```python
import spacy_trankit

# Initialize the pipeline
nlp = spacy_trankit.load_from_path(name="en", path="./cache") 

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```

## 📦 Model downloads

The Trankit release on PyPI fetches its pretrained models from
`nlp.uoregon.edu`, which is currently unavailable. `spacy-trankit` bypasses
that broken download path and pulls the same artifacts from Trankit's
HuggingFace mirror (`https://huggingface.co/uonlp/trankit`) into the local
cache before instantiating the Trankit pipeline. The behaviour is automatic;
no extra setup is needed.

If you mirror the artifacts elsewhere (e.g. for offline / air-gapped use),
point `spacy-trankit` at it via the `SPACY_TRANKIT_MODEL_URL` environment
variable. The template understands `{version}`, `{embedding}` and `{lang}`:

```bash
export SPACY_TRANKIT_MODEL_URL="https://my-mirror.example.com/trankit/{version}/{embedding}/{lang}.zip"
```
